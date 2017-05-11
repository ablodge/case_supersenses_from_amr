import os, statistics
from sklearn import svm, tree
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

os.chdir("C:\\Users\\Austin\\Desktop\\Project")


# te_file = "amr-bi_and_ngrams1.3-test.mff"


def get_data(f):
    data = {'examples': [], 'target': [], 'name': "", 'attributes': []}

    with open(f, "r") as fin:
        for line in fin:
            if line.startswith("@dataset "):
                data["name"] = line.split()[1]
            elif line.startswith("@attribute "):
                s = line.split()
                data["attributes"].append(s[1:])
            elif line.startswith("@examples") or line.strip() == "":
                continue
            else:
                s = line.split()
                datum = []
                for i, a in enumerate(s):
                    attr = data['attributes'][i]
                    if attr[1] == 'numeric':
                        datum.append(float(a))
                    else:
                        datum.append(attr.index(a) - 1)
                data['examples'].append(datum[:-1])
                data['target'].append(datum[-1])

    return data


def combine(data1, data2):
    data = {'examples': [], 'target': [], 'name': "", 'attributes': []}
    if len(data1['examples']) != len(data2['examples']):
        print(len(data1['examples']))
        print(len(data2['examples']))
    if data1['target'] != data2['target']:
        return None
    data['name'] = data1['name'] + ":" + data2['name']
    data['attributes'] = data1['attributes'][:-1] + data2['attributes']
    data['examples'] = [data1['examples'][i] + data2['examples'][i] for i in range(len(data1['examples']))]
    data['target'] = data1['target']
    return data


def main():

    # tr_files = ["pos1.3-train.mff","ngrams1.3-train.mff", "amrgrams0-train.mff", "pos_amr1.3-train.mff",
    #             "amr_and_ngrams1.3-train.mff","pos_amr_ngram1.3-train.mff"]
    # te_files = ["pos1.3-test.mff", "ngrams1.3-test.mff", "amrgrams0-test.mff", "pos_amr1.3-test.mff",
    #             "amr_and_ngrams1.3-test.mff", "pos_amr_ngram1.3-test.mff"]

    tr_files = ["pos_amr_ngram1.3-train.mff"]
    te_files = ["pos_amr_ngram1.3-test.mff"]

    clf = LogisticRegression()
    print(clf)
    for f_index in range(len(tr_files)):
        tr_data = get_data(tr_files[f_index])
        te_data = get_data(te_files[f_index])

        to_be_removed = []
        i = 0
        for s in tr_data['target']:
            if tr_data['attributes'][-1][s + 1] in ["Material","Co-Participant","ValueComparison","ClockTimeCxn"]:
                to_be_removed.append(i)
            i += 1

        for index in sorted(to_be_removed, reverse=True):
            del tr_data['target'][index]
            del tr_data['examples'][index]
        to_be_removed = []
        i = 0
        for s in te_data['target']:
            if te_data['attributes'][-1][s + 1] in ["Material", "Co-Participant", "ValueComparison", "ClockTimeCxn"]:
                to_be_removed.append(i)
            i += 1
        for index in sorted(to_be_removed, reverse=True):
            del te_data['target'][index]
            del te_data['examples'][index]

        # for s in supersenses:
        #     print(tr_data['attributes'][-1][s + 1] + " " + str(supersenses[s]))

        # clf = svm.SVC(C=1.0)

        clf.fit(tr_data['examples'],tr_data['target'])

        pred = clf.predict(te_data['examples'])
        conf = [[0 for x in range(63)] for y in range(63)]
        perf = 0
        for i in range(len(pred)):
            conf[pred[i]][te_data['target'][i]]+=1
            if pred[i] == te_data['target'][i]:
                perf+=1
        perf/=len(pred)
        print(tr_files[f_index])
        print(perf)
        print(conf)
        print(te_data['attributes'][-1])

        file = "log.txt"

        with open("log.txt", "a") as fout:
            fout.write(str(clf) + "\n")
            fout.write(tr_files[f_index]+"\n")
            fout.write(str(perf)+"\n\n")

    print("\a")

if __name__ == "__main__":
    main()
