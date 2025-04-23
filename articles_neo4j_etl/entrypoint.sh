#!/bin/bash

# Run any setup steps or pre-processing tasks here
echo "Running ETL to move article data from csvs to Neo4j..."

# Run the ETL script
python src/articles_neo4j_etl/article_bulk_csv_write.py