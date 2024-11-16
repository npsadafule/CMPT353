import sys
from pyspark.sql import SparkSession, functions, types

# Initialize Spark session
spark = SparkSession.builder.appName('Popular Wikipedia Pages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

# Schema for the Wikipedia page counts data
data_schema = types.StructType([
    types.StructField('language', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('requests', types.IntegerType()),
    types.StructField('bytes_returned', types.LongType()),
])

# Define a function to extract the date and hour from the file path
def extract_hour_from_path(path):
    """
    Extract the date and hour in the format 'YYYYMMDD-HH' from the file path.
    The filenames are in the format pagecounts-YYYYMMDD-HHMMSS eg. pagecounts-20160801-120000
    """
    filename = path.split('/')[-1]  # Extract the filename from the path, Ref: https://www.geeksforgeeks.org/python-program-to-get-the-file-name-from-the-file-path/
    print(filename)
    return filename[11:22]  # Extract 'YYYYMMDD-HH' from the filename

# Define the UDF to convert file paths to hour labels
path_to_hour = functions.udf(extract_hour_from_path, returnType=types.StringType())

def main(in_directory, out_directory):
    # Read the data from the input directory, adding the filename column
    pagecount = spark.read.csv(in_directory, schema=data_schema, sep=' ').withColumn('filename', functions.input_file_name())

    # Add a new column for the hour extracted from the filename
    pages_with_hour = pagecount.withColumn('hour', path_to_hour(functions.col('filename')))

    # Filter to include only English Wikipedia pages, excluding 'Main_Page' and 'Special:'
    filtered_pages = pages_with_hour.filter(
        (pages_with_hour['language'] == 'en') &
        (pages_with_hour['title'] != 'Main_Page') &
        (~pages_with_hour['title'].startswith('Special:'))
    ).cache()

    # Finding the maximum number of requests for each hour
    max_requests_per_hour = filtered_pages.groupBy('hour').agg(functions.max('requests').alias('max_requests'))

    # Storing in aliased dataset to avoid confusion
    filtered_pages_alias = filtered_pages.alias("fp")
    max_requests_per_hour_alias = max_requests_per_hour.alias("mr")

    # Join back to get the titles with the maximum number of requests per hour
    #ref: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.join.html
    most_viewed_pages = filtered_pages_alias.join(
        max_requests_per_hour_alias,
        (filtered_pages_alias['hour'] == max_requests_per_hour_alias['hour']) &
        (filtered_pages_alias['requests'] == max_requests_per_hour_alias['max_requests'])
    ).select(filtered_pages_alias['hour'], filtered_pages_alias['title'], filtered_pages_alias['requests'])

    # Sort the results by hour and title (to handle ties)
    sorted_results = most_viewed_pages.orderBy('hour', 'title')

    # Write the output to CSV
    sorted_results.write.csv(out_directory, mode='overwrite')

if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
