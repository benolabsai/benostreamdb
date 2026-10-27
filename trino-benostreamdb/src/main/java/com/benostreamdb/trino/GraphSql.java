package com.benostreamdb.trino;

import io.trino.spi.function.table.Descriptor;
import io.trino.spi.type.BigintType;
import io.trino.spi.type.IntegerType;
import io.trino.spi.type.Type;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * SQL and schema helpers for the single-edge-table graph functions.
 *
 * <p>The edge table is registered as {@code t} by {@code Table::sql}, so each
 * generated statement passes the literal {@code 't'} as the engine table
 * name.</p>
 */
public final class GraphSql {

    private static final Pattern SEEDS = Pattern.compile("[0-9]+(?:,[0-9]+)*");
    private static final Set<String> MODES = Set.of("auto", "in_memory", "out_of_core", "cached");

    private GraphSql() {
    }

    public static String build(GraphTableFunctionHandle handle) {
        Kind kind = Kind.fromName(handle.getFunction());
        // Validate the table reference even though the split carries its URI.
        BenoStreamDBTableUri.splitTable(handle.getTable());

        List<String> args = handle.getArgs();
        switch (kind) {
            case NEIGHBORS:
            case SUBGRAPH:
                requireArgs(args, 3);
                String seeds = validateSeeds(args.get(0));
                int hops = parseHops(args.get(1));
                String mode = validateMode(args.get(2));
                return "SELECT * FROM " + kind.functionName() + "('t', '" + seeds + "', "
                        + hops + ", '" + mode + "')";
            case SHORTEST_PATH:
                requireArgs(args, 3);
                long source = parseNode(args.get(0), "source");
                long target = parseNode(args.get(1), "target");
                String pathMode = validateMode(args.get(2));
                return "SELECT * FROM graph_shortest_path('t', " + source + ", "
                        + target + ", '" + pathMode + "')";
            default:
                throw new IllegalArgumentException("Unsupported graph function: " + kind);
        }
    }

    public static Descriptor descriptor(Kind kind) {
        List<String> names = new ArrayList<>();
        List<Type> types = new ArrayList<>();
        switch (kind) {
            case NEIGHBORS:
                names.add("node");
                types.add(BigintType.BIGINT);
                names.add("hop");
                types.add(IntegerType.INTEGER);
                names.add("seed");
                types.add(BigintType.BIGINT);
                break;
            case SHORTEST_PATH:
                names.add("node");
                types.add(BigintType.BIGINT);
                names.add("hop");
                types.add(IntegerType.INTEGER);
                break;
            case SUBGRAPH:
                names.add("source");
                types.add(BigintType.BIGINT);
                names.add("target");
                types.add(BigintType.BIGINT);
                break;
            default:
                throw new IllegalArgumentException("Unsupported graph function: " + kind);
        }
        return Descriptor.descriptor(names, types);
    }

    public static List<BenoStreamDBColumnHandle> columns(Kind kind) {
        List<BenoStreamDBColumnHandle> columns = new ArrayList<>();
        for (Descriptor.Field field : descriptor(kind).getFields()) {
            columns.add(new BenoStreamDBColumnHandle(
                    field.getName().orElseThrow(),
                    field.getType().orElseThrow()));
        }
        return columns;
    }

    private static void requireArgs(List<String> args, int count) {
        if (args == null || args.size() != count) {
            throw new IllegalArgumentException(
                    "Expected " + count + " arguments, got " + (args == null ? 0 : args.size()));
        }
    }

    private static String validateSeeds(String value) {
        if (value == null || !SEEDS.matcher(value).matches()) {
            throw new IllegalArgumentException(
                    "seeds must be a comma-separated list of unsigned integers");
        }
        return value;
    }

    private static int parseHops(String value) {
        try {
            int hops = Integer.parseInt(value);
            if (hops < 0 || hops > 1_000) {
                throw new IllegalArgumentException("hops must be between 0 and 1000");
            }
            return hops;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("hops must be an integer", e);
        }
    }

    private static long parseNode(String value, String label) {
        try {
            long node = Long.parseLong(value);
            if (node < 0) {
                throw new IllegalArgumentException(label + " must be non-negative");
            }
            return node;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(label + " must be an integer", e);
        }
    }

    private static String validateMode(String value) {
        String mode = value == null ? "auto" : value.trim().toLowerCase(Locale.ROOT);
        if (!MODES.contains(mode)) {
            throw new IllegalArgumentException(
                    "mode must be one of auto, in_memory, out_of_core, cached");
        }
        return mode;
    }

    public enum Kind {
        NEIGHBORS("graph_neighbors"),
        SHORTEST_PATH("graph_shortest_path"),
        SUBGRAPH("graph_subgraph");

        private final String functionName;

        Kind(String functionName) {
            this.functionName = functionName;
        }

        public String functionName() {
            return functionName;
        }

        public static Kind fromName(String name) {
            for (Kind kind : values()) {
                if (kind.functionName.equals(name)) {
                    return kind;
                }
            }
            throw new IllegalArgumentException("Unsupported graph function: " + name);
        }
    }
}
