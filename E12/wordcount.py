from pyspark.sql import SparkSession, functions, types
import sys, re, string

def main(input, output):
    lines = spark.read.text(input)
    #lines.show()

    #split the lines into words based on spaces and punctuation
    wordbreak = r'[%s\s]+' % (re.escape(string.punctuation)) 

    #functions.split() to split the value column into arrays of words
    #functions.explode() to transform the array of words into individual rows
    words = lines.select(functions.explode(functions.split(lines['value'], wordbreak)).alias('word'))
    #words.show()

    #Convert all words to lowercase to avoid counting the same word differently due to case differences
    words_lower = words.select(functions.lower(words['word']).alias('word'))
    #words_lower.show()

    #Remove empty strings that may result from splitting
    words_filtered = words_lower.filter(words_lower['word'] != '')

    #Group the DataFrame by the word column and count the number of occurrences
    word_counts = words_filtered.groupBy('word').count()
    #word_counts.show()
    
    #Sort the DataFrame by the count column in descending order and then in alphabetical order to avoid tie
    words_count_sorted = word_counts.orderBy(functions.desc('count'), 'word')
    #words_count_sorted.show()

    #Write the sorted DataFrame to a CSV file in the output directory
    words_count_sorted.write.csv(output, mode='overwrite', header=False)


if __name__ == '__main__':
    input = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName('wordcount').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(input, output)
