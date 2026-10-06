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
        if (type instanceof io.trino.spi.type.ArrayType) {
            return "List(" + arrowTypeName(((io.trino.spi.type.ArrayType) type).getElementType()) + ")";
        }
        if (type instanceof io.trino.spi.type.RowType) {
            StringBuilder sb = new StringBuilder("Struct(");
            List<io.trino.spi.type.RowType.Field> fields = ((io.trino.spi.type.RowType) type).getFields();
            for (int i = 0; i < fields.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(fields.get(i).getName().orElse("col" + i)).append(": ").append(arrowTypeName(fields.get(i).getType()));
            }
            return sb.append(")").toString();
        }
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
            fields.add(toArrowField(c.getColumnName(), c.getColumnType()));
        }
        return new Schema(fields);
    }

    private static Field toArrowField(String name, Type type) {
        if (type instanceof io.trino.spi.type.ArrayType) {
            Field child = toArrowField("item", ((io.trino.spi.type.ArrayType) type).getElementType());
            return new Field(name, FieldType.nullable(new ArrowType.List()), List.of(child));
        }
        if (type instanceof io.trino.spi.type.RowType) {
            List<Field> children = new ArrayList<>();
            List<io.trino.spi.type.RowType.Field> fields = ((io.trino.spi.type.RowType) type).getFields();
            for (int i = 0; i < fields.size(); i++) {
                children.add(toArrowField(fields.get(i).getName().orElse("col" + i), fields.get(i).getType()));
            }
            return new Field(name, FieldType.nullable(new ArrowType.Struct()), children);
        }
        return new Field(name, FieldType.nullable(arrowType(type)), null);
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
        if (type instanceof io.trino.spi.type.ArrayType) {
            org.apache.arrow.vector.complex.ListVector listVector = (org.apache.arrow.vector.complex.ListVector) vector;
            io.trino.spi.block.Block arrayBlock = ((io.trino.spi.type.ArrayType) type).getObject(block, pos);
            int startOffset = listVector.startNewValue(out);
            int len = arrayBlock.getPositionCount();
            Type elementType = ((io.trino.spi.type.ArrayType) type).getElementType();
            FieldVector childVector = listVector.getDataVector();
            for (int i = 0; i < len; i++) {
                if (arrayBlock.isNull(i)) {
                    childVector.setNull(startOffset + i);
                } else {
                    writeValue(childVector, elementType, arrayBlock, i, startOffset + i);
                }
            }
            listVector.endValue(out, len);
        } else if (type instanceof io.trino.spi.type.RowType) {
            org.apache.arrow.vector.complex.StructVector structVector = (org.apache.arrow.vector.complex.StructVector) vector;
            // Trino 435: RowType.getObject returns a SqlRow, not a Block.
            io.trino.spi.block.SqlRow row = ((io.trino.spi.type.RowType) type).getObject(block, pos);
            structVector.setIndexDefined(out);
            List<io.trino.spi.type.RowType.Field> fields = ((io.trino.spi.type.RowType) type).getFields();
            int rawIndex = row.getRawIndex();
            for (int i = 0; i < fields.size(); i++) {
                FieldVector childVector = structVector.getChild(fields.get(i).getName().orElse("col" + i));
                io.trino.spi.block.Block fieldBlock = row.getRawFieldBlock(i);
                if (fieldBlock.isNull(rawIndex)) {
                    childVector.setNull(out);
                } else {
                    writeValue(childVector, fields.get(i).getType(), fieldBlock, rawIndex, out);
                }
            }
        } else if (type instanceof IntegerType) {
            ((IntVector) vector).setSafe(out, (int) type.getLong(block, pos));
        } else if (type instanceof BigintType) {
            ((BigIntVector) vector).setSafe(out, type.getLong(block, pos));
        } else if (type instanceof RealType) {
            ((Float4Vector) vector).setSafe(out, Float.intBitsToFloat((int) type.getLong(block, pos)));
        } else if (type instanceof DoubleType) {
            // DoubleType does not support getLong (AbstractType throws); read the
            // double directly.
            ((Float8Vector) vector).setSafe(out, type.getDouble(block, pos));
        } else if (type instanceof BooleanType) {
            ((BitVector) vector).setSafe(out, type.getBoolean(block, pos) ? 1 : 0);
        } else if (type instanceof DateType) {
            ((DateDayVector) vector).setSafe(out, (int) type.getLong(block, pos));
        } else if (type instanceof VarcharType) {
            // Trino 468 moved slice access off `Block`; read it through the type.
            ((VarCharVector) vector).setSafe(out, ((VarcharType) type).getSlice(block, pos).getBytes());
        } else {
            throw new UnsupportedOperationException("Unsupported Trino type: " + type);
        }
    }
}
