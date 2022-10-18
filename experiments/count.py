import json

f_list = [
"test_mixvegrice_r50_30kits_batch8_result_0.4.json",
"test_mixvegrice_r50_30kits_batch8_result_0.45.json",
"test_mixvegrice_r50_30kits_batch8_result_0.55.json",
"test_mixvegrice_r50_30kits_batch8_result_0.6.json",
"test_mixvegrice_r50_30kits_batch8_result_0.65.json",
"test_mixvegrice_r50_30kits_batch8_result_0.7.json",
"test_mixvegrice_r50_20kits_batch8_result_0.4.json",
"test_mixvegrice_r50_20kits_batch8_result_0.45.json",
"test_mixvegrice_r50_20kits_batch8_result_0.55.json",
"test_mixvegrice_r50_20kits_batch8_result_0.6.json",
"test_mixvegrice_r50_20kits_batch8_result_0.65.json",
"test_mixvegrice_r50_20kits_batch8_result_0.7.json",
]

for n in f_list:
    data = json.load(open(n,'r'))
    pq = 0.0
    f1 = 0.0
    count = 0
    for key, val in data.items():
        if key != 'overall':
            pq += val.get('pq',0.0)
            f1 += val.get('f1',0.0)
            count += 1
    print(n)
    print("{:.2f},{:.2f}".format(pq/count * 100, f1/count * 100))