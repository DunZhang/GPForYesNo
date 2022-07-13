"""

把原始数据转换成训练用的数据

"""
import json
from os.path import join

if __name__ == "__main__":
    data_dir = r"F:\BaiduNetdiskDownload\DFData"
    save_dir = "./data_model"
    # 先读一遍获取所有的docid
    all_docid = set()
    for name in ["train", "dev"]:
        with open(join(data_dir, "News2022_task2_{}.tsv".format(name)), "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.strip().split("\t")
                item_id, query, ans_type, doc_id, answer = ss
                all_docid.add(doc_id)
    # 节省内存，值读取用到的doc
    docid2doc = {}
    with open(join(data_dir, "News2022_doc_B.tsv"), "r", encoding="utf8") as fr:
        for line in fr:
            doc_id, doc = line.strip().split("\t")
            if doc_id in all_docid:
                docid2doc[doc_id] = doc
    # 获取最终结果
    for name in ["train", "dev"]:
        save_data = []
        with open(join(data_dir, "News2022_task2_{}.tsv".format(name)), "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.strip().split("\t")
                item_id, query, ans_type, doc_id, answer = ss
                save_data.append({
                    "query": query,
                    "doc": docid2doc[doc_id],
                    "ans": answer,
                    "ans_type": ans_type.lower(),
                })
        with open(join(save_dir, "{}.json".format(name)), "w", encoding="utf8") as fw:
            json.dump(save_data, fw, ensure_ascii=False, indent=1)
