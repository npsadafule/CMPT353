import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, functions, types

comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('link_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('parent_id', types.StringType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('score_hidden', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('ups', types.LongType()),
    #types.StructField('year', types.IntegerType()),
    #types.StructField('month', types.IntegerType()),
])


def main(in_directory, out_directory):
    comments = spark.read.json(in_directory, schema=comments_schema)

    #Calculate the average score for each subreddit
    subreddit_avg = comments.groupBy('subreddit').agg(
        functions.avg('score').alias('avg_score')
    ).filter(functions.col('avg_score') > 0)

    # Cache the average score DataFrame
    #subreddit_avg = subreddit_avg.cache().hint('broadcast')
    subreddit_avg = subreddit_avg

    #Join the average score to the comments to calculate the relative score
    comments_alias = comments.alias('c')
    subreddit_avg_alias = subreddit_avg.alias('a')

    comments_with_avg = comments_alias.join(
        subreddit_avg_alias,
        on='subreddit'
    ).withColumn('rel_score', functions.col('c.score') / functions.col('a.avg_score'))

    #Determine the maximum relative score for each subreddit
    max_rel_score_per_subreddit = comments_with_avg.groupBy('subreddit').agg(
        functions.max('rel_score').alias('max_rel_score')
    )

    # Cache the max relative score DataFrame
    #max_rel_score_per_subreddit = max_rel_score_per_subreddit.cache().hint('broadcast')
    max_rel_score_per_subreddit = max_rel_score_per_subreddit

    #Join again to get the author of the best comment in each subreddit
    comments_with_avg_alias = comments_with_avg.alias('c')
    max_rel_score_alias = max_rel_score_per_subreddit.alias('m')

    best_author = comments_with_avg_alias.join(
        max_rel_score_alias,
        on=[
            comments_with_avg_alias['subreddit'] == max_rel_score_alias['subreddit'],
            comments_with_avg_alias['rel_score'] == max_rel_score_alias['max_rel_score']
        ]
    ).select(
        comments_with_avg_alias['subreddit'],
        comments_with_avg_alias['author'],
        comments_with_avg_alias['rel_score']
    )

    # Write the output in JSON
    best_author.write.json(out_directory, mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    spark = SparkSession.builder.appName('Reddit Relative Scores').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory, out_directory)
