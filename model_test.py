import unittest
import requests
import json

class APITest(unittest.TestCase):
    URL = "http://localhost:5000/api/predict"
    DATA = {
        "title" : "ヌートリアを岐阜大学で発見"
    }

    def test_normal_input(self):
        # リクエストを投げる
        response = requests.post(self.URL, json=self.DATA)
        # 結果
        print(response.text)  # 本来は不要だが，確認用
        result = json.loads(response.text)  # JSONをdictに変換
        # ステータスコードが200かどうか
        self.assertEqual(response.status_code, 200)
        # statusはOKかどうか
        self.assertEqual(result["status"], "OK")
        # 非負の予測値があるかどうか
        #self.assertTrue(0 <= result["predicted"])


if __name__ == "__main__":
    unittest.main()