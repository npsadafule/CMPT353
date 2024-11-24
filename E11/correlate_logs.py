import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types, Row
import re


line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred.
    Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        hostname = m.group(1)
        bytes_transferred = int(m.group(2))
        return Row(hostname=hostname, bytes=bytes_transferred)
    else:
        return None


def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    row_rdd = log_lines.map(line_to_row).filter(not_none)
    return row_rdd

def main(in_directory):
    #Create DataFrame from RDD
    logs = spark.createDataFrame(create_row_rdd(in_directory))

    #Group by hostname and calculate count of requests and sum of bytes transferred
    grouped_logs = logs.groupBy('hostname').agg(
        functions.count('*').alias('count_requests'),
        functions.sum('bytes').alias('sum_request_bytes')
    )

    #Calculate the six sums needed for correlation
    sums = grouped_logs.select(
        functions.sum('count_requests').alias('sum_x'),
        functions.sum('sum_request_bytes').alias('sum_y'),
        functions.sum(functions.pow('count_requests', 2)).alias('sum_x_sq'),
        functions.sum(functions.pow('sum_request_bytes', 2)).alias('sum_y_sq'),
        functions.sum(functions.col('count_requests') * functions.col('sum_request_bytes')).alias('sum_xy'),
        functions.count('*').alias('n')
    ).first()

    #Extract sums for correlation calculation
    sum_x = sums['sum_x']
    sum_y = sums['sum_y']
    sum_x_sq = sums['sum_x_sq']
    sum_y_sq = sums['sum_y_sq']
    sum_xy = sums['sum_xy']
    n = sums['n']

    #Calculate the correlation coefficient r
    numerator = (n * sum_xy) - (sum_x * sum_y)
    denominator = ((n * sum_x_sq - sum_x ** 2) ** 0.5) * ((n * sum_y_sq - sum_y ** 2) ** 0.5)
    r = numerator / denominator if denominator != 0 else 0

    # Print the result
    print(f"r = {r}\nr^2 = {r * r}")

    # Built-in function should get the same results (for verification)
    print(grouped_logs.corr('count_requests', 'sum_request_bytes'))

if __name__ == '__main__':
    in_directory = sys.argv[1]
    spark = SparkSession.builder.appName('correlate logs').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory)
