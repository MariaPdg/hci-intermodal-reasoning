import json


def read_caption():
    with open("dataset/annotations/captions_val2014.json") as json_file:
        data = json.load(json_file)

        id2cap = {}
        for ann in data["annotations"]:
            if ann["id"] not in id2cap:
                id2cap[ann["id"]] = [ann["caption"]]
            else:
                id2cap[ann["id"]].append(ann["caption"])

        filename2id = {}
        for img in data["images"]:
            assert img["file_name"] not in filename2id
            filename2id[img["file_name"]] = [img["id"]]

    return id2cap, filename2id


if __name__ == "__main__":
    read_caption()
