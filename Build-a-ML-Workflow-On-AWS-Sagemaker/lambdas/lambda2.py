"""
ImageClassifier : Lambda function to predict image classification
"""
import os
import io
import boto3
import json
import base64

# setting the  environment variables
ENDPOINT_NAME = 'image-classification-2023-10-24-20-45-12-512'
# We will be using the AWS's lightweight runtime solution to invoke an endpoint.
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    
    # Decode the image data
    image = base64.b64decode(event["body"]["image_data"])
    
    # Make a prediction:
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='image/png',
                                       Body=image)
    
    # We return the data back to the Step Function    
    event["body"]["inferences"] = json.loads(response['Body'].read().decode('utf-8'))
    return {
        'statusCode': 200,
        'body': event["body"]
    }
