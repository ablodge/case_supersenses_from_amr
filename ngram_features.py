import os

train_dir = "C:\\Users\\Austin\\Desktop\\Project\\train"
test_dir = "C:\\Users\\Austin\\Desktop\\Project\\test"
file2 = "C:\\Users\\Austin\\Desktop\\Project\\psst-tokens.tsv"
file3 = "C:\\Users\\Austin\\Desktop\\Project\\psst-test.sentids"

train_dataset = []
test_dataset = []
sentences = {}
test_ids = []

ID = 0
PREP = 1
ANNO = 2
SENT = 3
SENT2 = 4

D = 6
N = 3

train_output = "C:\\Users\\Austin\\Desktop\\Project\\ngrams"+str(D)+"."+str(N)+"-train.mff"
test_output = "C:\\Users\\Austin\\Desktop\\Project\\ngrams"+str(D)+"."+str(N)+"-test.mff"


# get sentences
i = 0
while os.path.isfile(os.path.join(train_dir, "train" + str(i) + ".txt")):
    tr = os.path.join(train_dir, "train" + str(i) + ".txt")
    id = os.path.join(train_dir, "id" + str(i) + ".txt")
    with open(tr, "rt") as fin1:
        with open(id, "rt") as fin2:
            for line1, line2 in zip(fin1, fin2):
                sentences[line2.strip()] = line1.strip()
    i += 1
i = 0
while os.path.isfile(os.path.join(test_dir, "test" + str(i) + ".txt")):
    tr = os.path.join(test_dir, "test" + str(i) + ".txt")
    id = os.path.join(test_dir, "id" + str(i) + ".txt")
    with open(tr, "rt") as fin1:
        with open(id, "rt") as fin2:
            for line1, line2 in zip(fin1, fin2):
                sentences[line2.strip()] = line1.strip().lower()
    i += 1

# get datasets
with open(file3, "rt") as fin:
    for line in fin:
        test_ids.append(line.strip())
with open(file2, "rt") as fin:
    for line in fin:
        s = line.split("\t")
        id = s[ID].split(":")[ID]
        s.append(sentences[id])
        if id in test_ids:
            test_dataset.append(s)
        else:
            train_dataset.append(s)


def get_ngrams(t, index):
    tokens = ["<s>"]
    tokens.extend(t)
    tokens.append("</s>")
    index += 1
    start = index - D
    end = index + D + 1
    if start < 0:
        start = 0
    if end > len(tokens):
        end = len(tokens) - 1
    ngrams = get_ngrams_in_range(tokens[start:end])
    ngrams[:] = [n for n in ngrams if n != ""]
    return ngrams


def get_ngrams_in_range(token_span):
    ngrams = []
    ngrams.extend(token_span)
    if "" in ngrams:
        ngrams.remove("")
    if len(token_span) < 2 or N < 2:
        return ngrams
    for index in range(1,len(token_span)):
        ngrams.append(":".join(token_span[index - 1:index+1]))
    if len(token_span) < 3 or N < 3:
        return ngrams
    for index in range(2,len(token_span)):
        ngrams.append(":".join(token_span[index - 2:index+1]))
    if len(token_span) < 4 or N < 4:
        return ngrams
    for index in range(3,len(token_span)):
        ngrams.append(":".join(token_span[index - 3:index+1]))
    return ngrams


attribute_set = set()
class_attribute = set()
train_examples = []
test_examples = []

for datum in train_dataset:
    if " " in datum[ANNO]:
        continue
    sentence = datum[SENT].split()
    j = next(i for i in range(len(sentence)) if sentence[i].startswith("|"))
    sentence = datum[SENT2].lower().split()
    ngrams = get_ngrams(sentence, j)
    for ng in ngrams:
        attribute_set.add(ng)
    class_attribute.add(datum[ANNO])
for datum in test_dataset:
    class_attribute.add(datum[ANNO])

attributes = []
attributes.extend(attribute_set)
attributes.append("<UNK>")

for datum in train_dataset:
    if " " in datum[PREP]:
        continue
    sentence = datum[SENT].split()
    j = next(i for i in range(len(sentence)) if sentence[i].startswith("|"))
    sentence = datum[SENT2].lower().split()
    ngrams = get_ngrams(sentence, j)
    train_examples.append(["0" for i in range(len(attributes))])
    for ng in ngrams:
        try:
            i = attributes.index(ng)
        except IndexError:
            i = -1
        train_examples[-1][i] = "1"
    train_examples[-1].append(datum[ANNO])

for datum in test_dataset:
    if " " in datum[PREP]:
        continue
    sentence = datum[SENT].split()
    j = next(i for i in range(len(sentence)) if sentence[i].startswith("|"))
    sentence = datum[SENT2].lower().split()
    ngrams = get_ngrams(sentence, j)
    test_examples.append(["0" for i in range(len(attributes))])
    for ng in ngrams:
        try:
            i = attributes.index(ng)
        except ValueError:
            i = -1
        test_examples[-1][i] = "1"
    test_examples[-1].append(datum[ANNO])

# @dataset austin
#
# @attribute a1 y n
# @attribute a2 y n
# @attribute class + -
#
# @examples
#
# y y +
# y n +
# n y -
# n n -

with open(train_output, "wt") as fout:
    fout.write("@dataset ngrams\n")
    fout.write("\n")
    for a in attributes:
        fout.write("@attribute " + a + " 1 0\n")
    fout.write("@attribute class")
    for c in class_attribute:
        fout.write(" " + c)
    fout.write("\n\n")
    fout.write("@examples\n")
    fout.write("\n")
    for ex in train_examples:
        fout.write(ex[0])
        for i in ex[1:]:
            fout.write(" " + i)
        fout.write("\n")

with open(test_output, "wt") as fout:
    fout.write("@dataset ngrams\n")
    fout.write("\n")
    for a in attributes:
        fout.write("@attribute " + a + " 1 0\n")
    fout.write("@attribute class")
    for c in class_attribute:
        fout.write(" " + c)
    fout.write("\n\n")
    fout.write("@examples\n")
    fout.write("\n")
    for ex in test_examples:
        fout.write(ex[0])
        for i in ex[1:]:
            fout.write(" " + i)
        fout.write("\n")

print(str(D)+"."+str(N))
print("examples " + str(len(train_examples)) + " " + str(len(test_examples)))
print("class attr " + str(len(class_attribute)))
print("features " + str(len(attributes)))
