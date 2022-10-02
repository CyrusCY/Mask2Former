import json

json_data = json.load(open('test_uec_result_0.5.json'))

for key, val in json_data.items():
    if key != "overall":
        if val.get('pq') > 0.9:
            print(f"0.9pq{key}")
        # if val.get('f1') > 0.9:
        #     print(f"0.9f1{key}")
        # if val.get('pq') < 0.3:
        #     print(f"0.3pq{key}")
        # if val.get('f1') < 0.3:
        #     print(f"0.3f1{key}")