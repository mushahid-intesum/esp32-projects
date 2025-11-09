from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Small BYOL head dimensions
EMBED_DIM = 64
PROJ_DIM = 32

# Global aggregation (simple averaging)
global_proj_w = np.random.randn(PROJ_DIM, EMBED_DIM) * 0.1
global_proj_b = np.zeros(PROJ_DIM)
global_pred_w1 = np.random.randn(PROJ_DIM, PROJ_DIM) * 0.1
global_pred_w2 = np.random.randn(PROJ_DIM, PROJ_DIM) * 0.1
global_pred_b1 = np.zeros(PROJ_DIM)
global_pred_b2 = np.zeros(PROJ_DIM)

@app.route("/weights", methods=["POST", "GET"])
def weights():
    global global_proj_w, global_proj_b
    global global_pred_w1, global_pred_w2
    global global_pred_b1, global_pred_b2

    if request.method == "POST":
        # Deserialize float array
        payload = request.data
        arr = np.frombuffer(payload, dtype=np.float32)
        idx = 0

        print(payload)
        print(arr)

        # Load proj
        for i in range(PROJ_DIM):
            global_proj_b[i] = arr[idx]
            idx += 1
            for j in range(EMBED_DIM):
                global_proj_w[i,j] = arr[idx]
                idx += 1
        # Load predictor
        for i in range(PROJ_DIM):
            global_pred_b1[i] = arr[idx]
            idx += 1
            global_pred_b2[i] = arr[idx]
            idx += 1
            for j in range(PROJ_DIM):
                global_pred_w1[i,j] = arr[idx]
                idx += 1
                global_pred_w2[i,j] = arr[idx]
                idx += 1

        # Optionally: aggregate from multiple clients here
        print("Received weights from a client.")
        return "OK"

    elif request.method == "GET":
        # Serialize and send global weights
        arr = []
        for i in range(PROJ_DIM):
            arr.append(global_proj_b[i])
            for j in range(EMBED_DIM):
                arr.append(global_proj_w[i,j])
        for i in range(PROJ_DIM):
            arr.append(global_pred_b1[i])
            arr.append(global_pred_b2[i])
            for j in range(PROJ_DIM):
                arr.append(global_pred_w1[i,j])
                arr.append(global_pred_w2[i,j])

        print("Sent weights to a client.")
        # BUG: null is being returned
        ret = np.array(arr, dtype=np.float32)
        print(ret)

        # ret = ret.tobytes()

        return ret

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
