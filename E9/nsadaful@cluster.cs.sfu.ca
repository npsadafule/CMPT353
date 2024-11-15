import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('weather ETL').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

observation_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.StringType()),
    types.StructField('observation', types.StringType()),
    types.StructField('value', types.IntegerType()),
    types.StructField('mflag', types.StringType()),
    types.StructField('qflag', types.StringType()),
    types.StructField('sflag', types.StringType()),
    types.StructField('obstime', types.StringType()),
])


def main(in_directory, out_directory):

    weather = spark.read.csv(in_directory, schema=observation_schema)

    # Filtering the data based on records we care about
    filtered_data = weather.filter(
        (functions.col('qflag').isNull()) &  # Quality flag must be null
        (functions.col('station').startswith('CA')) &  # Station must start with 'CA'
        (functions.col('observation') == 'TMAX')  # Observation type must be 'TMAX'
    )

    # Change temperature value to Celsius and rename it as tmax
    cleaned_data = filtered_data.select(
        functions.col('station'),
        functions.col('date'),
        (functions.col('value') / 10.0).alias('tmax')  # Convert to °C
    )

    cleaned_data.write.json(out_directory, compression='gzip', mode='overwrite')


if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
