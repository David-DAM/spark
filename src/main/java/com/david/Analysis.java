package com.david;

import com.david.domain.Employee;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.apache.spark.sql.functions.*;

public class Analysis {

    public static Dataset<Row> countEveryLetter(Dataset<Row> df) {
        return df.select(explode(split(df.col("city"), "")).as("letter"))
                .groupBy("letter")
                .count()
                .where("letter != ''")
                .orderBy(desc("count"));
    }

    public static Dataset<Row> averageSalary(Dataset<Row> df) {
        return df.select(format_number(avg("salary"), 2).as("Average salary"));
    }

    public static Dataset<Row> countDistinctCities(Dataset<Row> df) {
        return df.select(count_distinct(df.col("city")).as("Distinct city count"));
    }

    public static Dataset<Row> youngPeopleWithHighSalary(Dataset<Row> df) {
        return df.where("salary > 100000")
                .where("age < 30");
    }

    public static Dataset<Row> maxAndMinSalary(Dataset<Row> df) {
        return df.agg(max("salary"), min("salary"));
    }

    public static Dataset<Row> maxSalaryByCity(Dataset<Row> df) {

        return df.groupBy("city").max("salary");
    }

    public static void rddGroupBy(JavaSparkContext sparkContext, String csvFilePath) {
        JavaRDD<Employee> employeeJavaRDD = sparkContext.textFile(csvFilePath).map(line -> {
            String[] fields = line.split(",");
            Employee employee = new Employee();
            employee.setId(Long.valueOf(fields[0]));
            employee.setName(fields[1]);
            employee.setAge(Integer.parseInt(fields[2]));
            employee.setCity(fields[3]);
            employee.setSalary(Double.parseDouble(fields[4]));
            return employee;
        });

        JavaPairRDD<String, Iterable<Employee>> employeeByCities = employeeJavaRDD.groupBy(Employee::getCity);

        employeeByCities.foreach(tuple2 -> {
            System.out.println("City: " + tuple2._1);
            tuple2._2().forEach(System.out::println);
        });
    }

    public static void parallelizeCollection(JavaSparkContext sparkContext) {
        List<Integer> list = Stream.iterate(0, i -> i + 1).limit(10).collect(Collectors.toUnmodifiableList());

        JavaRDD<Integer> parallelize = sparkContext.parallelize(list).filter(e -> e % 2 == 0);
        System.out.println("Count of elements: " + parallelize.count());
    }
}
