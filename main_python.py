#!/usr/bin/env python3
"""
Create DynamoDB table for Solar Predictions
"""

import boto3
from botocore.exceptions import ClientError

def create_dynamodb_table():
    """Create the SolarPredictions table if it doesn't exist"""
    
    dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')
    
    try:
        # Check if table exists
        table = dynamodb.Table('SolarPredictions')
        table.load()
        print(f"✅ Table 'SolarPredictions' already exists")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            # Table doesn't exist, create it
            print("Creating table 'SolarPredictions'...")
            
            table = dynamodb.create_table(
                TableName='SolarPredictions',
                KeySchema=[
                    {
                        'AttributeName': 'request_id',
                        'KeyType': 'HASH'  # Partition key
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'request_id',
                        'AttributeType': 'S'
                    }
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            
            # Wait for table to be created
            table.meta.client.get_waiter('table_exists').wait(TableName='SolarPredictions')
            print(f"✅ Table 'SolarPredictions' created successfully")
            return True
        else:
            print(f"❌ Error: {e}")
            return False

if __name__ == "__main__":
    create_dynamodb_table()
