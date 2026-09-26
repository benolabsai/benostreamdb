package com.benostreamdb.trino;

import io.trino.spi.Page;
import io.trino.spi.block.Block;
import io.trino.spi.type.BigintType;
import io.trino.spi.type.BooleanType;
import io.trino.spi.type.DateType;
import io.trino.spi.type.DoubleType;
import io.trino.spi.type.IntegerType;
import io.trino.spi.type.RealType;
import io.trino.spi.type.Type;
import io.trino.spi.type.VarcharType;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.ArrayList;
import java.util.List;

/**
 * Converts Trino {@link Page}s into an Arrow {@link VectorSchemaRoot} for the
 * native write path. Supports the common scalar types; anything else is
 * rejected with a clear error rather than silently mis-encoded.
 */
public final class BenoStreamDBArrowConverter {

    private BenoStreamDBArrowConverter() {
    }

    /** The `[{name, type, nullable}]` JSON the native `createTable` expects. */
    public static String toSchemaJson(List<BenoStreamDBColumnHandle> columns) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < columns.size(); i++) {
            BenoStreamDBColumnHandle c = columns.get(i);
            if (i > 0) {
                sb.append(',');
            }
            sb.append("{\"name\":\"").append(c.getColumnName())
                    .append("\",\"type\":\"").append(arrowTypeName(c.getColumnType()))
                    .append("\",\"nullable\":true}");
        }
        return sb.append(']').toString();
    }

    private static String arrowTypeName(Type type) {
        if (type instanceof IntegerType) {
            return "Int32";
        }
        if (type instanceof BigintType) {
            return "Int64";
        }
        if (type instanceof RealType) {
            return "Float32";
        }
        if (type instanceof DoubleType) {
            return "Float64";
        }
        if (type instanceof BooleanType) {
            return "Boolean";
        }
        if (type instanceof DateType) {
            return "Date32";
        }
        if (type instanceof VarcharType) {
            return "Utf8";
        }
        throw new UnsupportedOperationException("Unsupported Trino type: " + type);
    }

    public static Schema toArrowSchema(List<BenoStreamDBColumnHandle> columns) {
        List<Field> fields = new ArrayList<>();
        for (BenoStreamDBColumnHandle c : columns) {
            fields.add(new Field(c.getColumnName(), FieldType.nullable(arrowType(c.getColumnType())), null));
        }
        return new Schema(fields);
    }

    private static ArrowType arrowType(Type type) {
        if (type instanceof IntegerType) {
            return new ArrowType.Int(32, true);
        }
        if (type instanceof BigintType) {
            return new ArrowType.Int(64, true);
        }
        if (type instanceof RealType) {
            return new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE);
        }
        if (type instanceof DoubleType) {
            return new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
        }
        if (type instanceof BooleanType) {
            return ArrowType.Bool.INSTANCE;
        }
        if (type instanceof DateType) {
            return new ArrowType.Date(DateUnit.DAY);
        }
        if (type instanceof VarcharType) {
            return ArrowType.Utf8.INSTANCE;
        }
        throw new UnsupportedOperationException("Unsupported Trino type for Arrow conversion: " + type);
    }

    public static VectorSchemaRoot toArrow(
            List<BenoStreamDBColumnHandle> columns, List<Page> pages, BufferAllocator allocator) {
        VectorSchemaRoot root = VectorSchemaRoot.create(toArrowSchema(columns), allocator);
        root.allocateNew();
        int row = 0;
        for (Page page : pages) {
            int positions = page.getPositionCount();
            for (int ch = 0; ch < columns.size(); ch++) {
                Type type = columns.get(ch).getColumnType();
                Block block = page.getBlock(ch);
                FieldVector vector = root.getVector(ch);
                for (int p = 0; p < positions; p++) {
                    if (block.isNull(p)) {
                        vector.setNull(row + p);
                    } else {
                        writeValue(vector, type, block, p, row + p);
                    }
                }
            }
            row += positions;
        }
        root.setRowCount(row);
        return root;
    }

    private static void writeValue(FieldVector vector, Type type, Block block, int pos, int out) {
        if (type instanceof IntegerType) {
            ((IntVector) vector).setSafe(out, (int) type.getLong(block, pos));
        } else if (type instanceof BigintType) {
            ((BigIntVector) vector).setSafe(out, type.getLong(block, pos));
        } else if (type instanceof RealType) {
            ((Float4Vector) vector).setSafe(out, Float.intBitsToFloat((int) type.getLong(block, pos)));
        } else if (type instanceof DoubleType) {
            ((Float8Vector) vector).setSafe(out, Double.longBitsToDouble(type.getLong(block, pos)));
        } else if (type instanceof BooleanType) {
            ((BitVector) vector).setSafe(out, type.getBoolean(block, pos) ? 1 : 0);
        } else if (type instanceof DateType) {
            ((DateDayVector) vector).setSafe(out, (int) type.getLong(block, pos));
        } else if (type instanceof VarcharType) {
            int len = block.getSliceLength(pos);
            ((VarCharVector) vector).setSafe(out, block.getSlice(pos, 0, len).getBytes());
        } else {
            throw new UnsupportedOperationException("Unsupported Trino type: " + type);
        }
    }
}
