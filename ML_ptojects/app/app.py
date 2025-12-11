from flask import Flask, render_template, request
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import VectorAssembler

app = Flask(__name__)

# ---------------------------------------------------
# Initialize Spark
# ---------------------------------------------------
spark = SparkSession.builder \
    .appName("FlaskSparkModel") \
    .getOrCreate()

# ---------------------------------------------------
# Load Spark ML Model
# ---------------------------------------------------
model_path = "infantheight2.model"        # CHANGE this to your model folder
model = PipelineModel.load(model_path)

# ---------------------------------------------------
# Prediction Function (your logic)
# ---------------------------------------------------
def predict_value(weight):
    assembler = VectorAssembler(inputCols=["weight"], outputCol="features")

    # Your original prediction data format
    data = [[float(weight), 0]]     # height = dummy 0
    columns = ["weight", "height"]

    df = spark.createDataFrame(data, columns)
    df2 = assembler.transform(df).select("features", "height")

    predictions = model.transform(df2).collect()
    return predictions[0]["prediction"]


# ---------------------------------------------------
# Routes
# ---------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    weight = request.form.get("weight")

    result = predict_value(weight)

    return render_template("home.html", result=result, input_weight=weight)


# ---------------------------------------------------
# Run Flask
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
