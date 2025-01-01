package com.david;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.graphx.Edge;
import org.apache.spark.graphx.Graph;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.List;

import static org.apache.spark.sql.functions.abs;
import static org.apache.spark.sql.functions.col;

public class Graphs {

    public static void salaryDifferenceGraphWithTopConnections(Dataset<Row> df) {
        // Create alias for self-join
        Dataset<Row> df1 = df.alias("e1");
        Dataset<Row> df2 = df.alias("e2");

        // Realize self-join and calculate the difference between salaries
        Dataset<Row> relationships = df1.join(df2, df1.col("id").notEqual(df2.col("id")))
                .withColumn("salary_diff",
                        abs(col("e1.salary").minus(col("e2.salary"))).divide(col("e1.salary"))
                )
                .filter(col("salary_diff").leq(0.1))
                .select(
                        col("e1.id").alias("employee1"),
                        col("e2.id").alias("employee2"),
                        col("salary_diff")
                );

        // Converter DataFrames to RDDs for GraphX
        JavaRDD<Tuple2<Object, String>> vertex = df.javaRDD().map(row ->
                new Tuple2<>(row.getInt(0), row.getString(1) + " (" + row.getString(3) + ")")
        );

        JavaRDD<Edge<String>> edges = relationships.javaRDD().map(row ->
                new Edge<>(
                        row.getInt(0), // employee1 ID
                        row.getInt(1), // employee2 ID
                        "similar_salary" // relation
                )
        );

        // Build the graph
        Graph<String, String> graph = Graph.apply(
                vertex.rdd(),
                edges.rdd(),
                "default",
                StorageLevel.MEMORY_AND_DISK(),
                StorageLevel.MEMORY_AND_DISK(),
                scala.reflect.ClassTag$.MODULE$.apply(String.class), // Class for vertex
                scala.reflect.ClassTag$.MODULE$.apply(String.class)  // Class for edges
        );

        // Analysis of the graph
        // 1. Number of de vertex y edges
        //System.out.println("Number of vertex: " + graph.vertices().count());
        //System.out.println("Number of edges: " + graph.edges().count());

        // 2. Employees with more connexions
        System.out.println("Top employees with more connections:");
        JavaPairRDD<Long, Integer> degrees = graph.edges()
                .toJavaRDD()
                .flatMapToPair(edge -> {
                    // Emit the vertex of origin and destiny for counting the grade of input and output
                    return List.of(new Tuple2<>(edge.srcId(), 1), new Tuple2<>(edge.dstId(), 1)).iterator();
                })
                .reduceByKey(Integer::sum);// Reduce by summing the connections for every vertex


        // Order by the grade (value) from higher to lower
        JavaPairRDD<Long, Integer> sortedDegrees = degrees
                .mapToPair(tuple -> new Tuple2<>(tuple._2, tuple._1)) // Change key and value to order by value
                .sortByKey(false) // Order descending by the grade
                .mapToPair(tuple -> new Tuple2<>(tuple._2, tuple._1)); // Change to the original format again

        // Take the 5 highest vertex
        sortedDegrees.take(5).forEach(degree ->
                System.out.println("Employee ID: " + degree._1 + ", Connections: " + degree._2)
        );
    }

}
