from flask import Flask, request, jsonify, abort
from flask_cors import CORS, cross_origin
import pandas as pd
import pickle
from janome.tokenizer import Tokenizer
from datetime import datetime
import sys
sys.path.append("./models")  # 前処理で使った自作モジュール「pipeline」を読み込むためPYTHONPATHに追加
app = Flask(__name__)
CORS(app, support_credentials=True)

# アプリ起動時に前処理パイプラインと予測モデルを読み込んでおく
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))
model = pickle.load(open("models/lgbm.pkl", "rb"))
dic = pickle.load(open("label2genre.pkl", "rb"))


@app.route('/api/predict', methods=["POST"])
@cross_origin(supports_credentials=True) 
def predict():
    """/api/predict にPOSTリクエストされたら予測値を返す関数"""
    try:
        response = request.headers
        # APIにJSON形式で送信された特徴量
        X = pd.DataFrame(request.json, index=[0])
        
        X = X["title"][0]
        # 前処理
        t = Tokenizer(wakati=True)
        X = " ".join([token for token in t.tokenize(X)])
        X = tfidf.transform([X])
        # 予測
        y_pred = model.predict(X)
        print(y_pred.argmax(1))
        pred = dic[int(y_pred.argmax(1)[0])]
        response = {"status": "OK", "predicted": pred}
        print(response)
        return jsonify(response), 200
    except Exception as e:
        print(e)  # デバッグ用
        abort(400)


@app.errorhandler(400)
def error_handler(error):
    """abort(400) した時のレスポンス"""
    response = {"status": "Error", "message": "Invalid Parameters"}
    return jsonify(response), error.code


if __name__ == "__main__":
    app.run()  # 開発用サーバーの起動