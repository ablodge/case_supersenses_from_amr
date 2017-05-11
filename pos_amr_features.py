import os

# original
train_dir = "C:\\Users\\Austin\\Desktop\\Project\\train"
test_dir = "C:\\Users\\Austin\\Desktop\\Project\\test"
psst_file = "C:\\Users\\Austin\\Desktop\\Project\\psst-tokens.tsv"
test_sentids_file = "C:\\Users\\Austin\\Desktop\\Project\\psst-test.sentids"

sentences = {} # id -> sentence string
test_ids = [] # from file psst-test.sentids
pos_list = {}

# JAMR parses
test_parse_dir = "C:\\Users\\Austin\\Desktop\\Project\\parse\\test-parse"
train_parse_dir = "C:\\Users\\Austin\\Desktop\\Project\\parse\\train-parse"

train_dataset = [] # list of data
test_dataset = [] # list of data
ID = 0
PREP = 1
ANNO = 2
SENT = 3
SENT2 = 4
NODES = 5  # from JAMR
EDGES = 6  # from JAMR
POS = 7

snt = []
nodes = []  # nodes[sentid][node][id,token,alignment]
edges = []  # nodes[sentid][edge][node1,label,node2,id1,id2]

N = 3
D = 1

train_output = "C:\\Users\\Austin\\Desktop\\Project\\pos_amr"+str(D)+"."+str(N)+"-train.mff"
test_output = "C:\\Users\\Austin\\Desktop\\Project\\pos_amr"+str(D)+"."+str(N)+"-test.mff"


pos_file = "C:\\Users\\Austin\\Desktop\\Project\\streusle.upos.tags"

with open(pos_file, "rt") as fin:
    for line in fin:
        s = line.split("\t")
        if len(s)<9:
            continue
        id = s[8].strip()
        if id not in pos_list:
            pos_list[id] = []
        pos = s[3]
        pos_list[id].append(pos)

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

print("sentences +%d" % len(sentences))

parse_files = []
parse_files.extend([os.path.join(train_parse_dir,f) for f in os.listdir(train_parse_dir)])
parse_files.extend([os.path.join(test_parse_dir,f) for f in os.listdir(test_parse_dir)])
for f in parse_files:
    if f.endswith("err"):
        continue
    with open(f, "rt") as fin1:
        for line in fin1:
            if line.startswith("# ::snt "):
                s = line[len("# ::snt "):].strip().lower()
                snt.append(s)
                nodes.append([])
                edges.append([])
            elif line.startswith("# ::node"):
                n = line[len("# ::node\t"):].strip().split("\t")
                nodes[-1].append(n)
            elif line.startswith("# ::edge"):
                e = line[len("# ::edge\t"):].strip().split("\t")
                edges[-1].append(e)

print("nodes +%d" % len(nodes))
print("edges +%d" % len(edges))

# get datasets
with open(test_sentids_file, "rt") as fin:
    for line in fin:
        test_ids.append(line.strip())
with open(psst_file, "rt") as fin:
    for line in fin:
        s = line.split("\t")
        sentid = s[ID].split(":")[ID]
        s.append(sentences[sentid].lower())
        i = snt.index(sentences[sentid].lower())
        s.append(nodes[i])
        s.append(edges[i])
        s.append(pos_list[sentid])
        if sentid in test_ids:
            test_dataset.append(s)
        else:
            train_dataset.append(s)

print("test data +%d" % len(test_dataset))
print("train data +%d" % len(train_dataset))

def index_node(datum, sent_index):
    ns = datum[NODES]
    i = 0
    for node in ns:
        # 2=alignment
        if node[2].startswith(str(sent_index) + "-"):
            return i
        i += 1
    return -1


def index_edge(datum, sent_index):
    for i in range(sent_index, len(datum[SENT])):
        j = index_node(datum, i)
        if j == -1:
            continue
        for k in range(len(datum[EDGES])):
            # edges 4=id2, nodes 0=id
            if datum[EDGES][k][4] == datum[NODES][j][0]:
                return k
    return -1


def get_amrgrams_node(datum, node_index):
    amr_grams = []
    edges_in = []
    edges_out = []
    # 1=node label
    amr_grams.append(datum[NODES][node_index][1])
    for e in datum[EDGES]:
        # 3=id1, 4=id2
        if e[3] == datum[NODES][node_index][0]:
            edges_out.append(e)
        if e[4] == datum[NODES][node_index][0]:
            edges_in.append(e)
    for e in edges_in:
        # 1=label
        amr_grams.append(e[1] + "-in")
        amr_grams.append(e[0])
        amr_grams.append(e[0] + ":" + e[1] + "-in")
    for e in edges_out:
        # 1=label
        amr_grams.append(e[1] + "-out")
        amr_grams.append(e[2])
        amr_grams.append(e[1] + "-out:" + e[2])
    # for i in range(len(amr_grams)):
    #     for j in range(len(amr_grams[i+1:])):
    #         if ":" in amr_grams[i] and ":" in amr_grams[j]:
    #             amr_grams.append(amr_grams[i]+":"+amr_grams[j])
    return amr_grams


def get_amrgrams_edge(datum, edge_index):
    n1 = 0
    n2 = 0
    i = 0
    for n in datum[NODES]:
        # nodes 0=id, edge 3=id1, 4=id2
        if n[0] == datum[EDGES][edge_index][3]:
            n1 = i
        if n[0] == datum[EDGES][edge_index][4]:
            n1 = i
        i += 1
    amr_grams = []
    amr_grams.extend(get_amrgrams_node(datum, n1))
    amr_grams.extend(get_amrgrams_node(datum, n2))
    amr_grams.append(datum[EDGES][edge_index][1])
    return amr_grams


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
    if " " in datum[PREP]:
        continue
    sentence = datum[SENT].split()
    sent_index = next(i for i in range(len(sentence)) if sentence[i].startswith("|"))
    amr_index = index_node(datum, sent_index)
    amrgrams = []
    if amr_index != -1:
        amrgrams[:] = get_amrgrams_node(datum, amr_index)
    else:
        amr_index = index_edge(datum, sent_index)
        if amr_index != -1:
            amrgrams[:] = get_amrgrams_edge(datum, amr_index)
        else:
            continue
    sentence = datum[POS]
    ngrams = get_ngrams(sentence, sent_index)
    amrgrams.extend(ngrams)
    for g in amrgrams:
        attribute_set.add(g)
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
    sent_index = next(i for i in range(len(sentence)) if sentence[i].startswith("|"))
    amr_index = index_node(datum, sent_index)
    amrgrams = []
    if amr_index != -1:
        amrgrams[:] = get_amrgrams_node(datum, amr_index)
    else:
        amr_index = index_edge(datum, sent_index)
        if amr_index != -1:
            amrgrams[:] = get_amrgrams_edge(datum, amr_index)
        else:
            continue
    sentence = datum[POS]
    ngrams = get_ngrams(sentence, sent_index)
    amrgrams.extend(ngrams)
    train_examples.append(["0" for i in range(len(attributes))])
    for g in amrgrams:
        try:
            i = attributes.index(g)
        except IndexError:
            i = -1
        train_examples[-1][i] = "1"
    train_examples[-1].append(datum[ANNO])

for datum in test_dataset:
    if " " in datum[PREP]:
        continue
    sentence = datum[SENT].split()
    sent_index = next(i for i in range(len(sentence)) if sentence[i].startswith("|"))
    amr_index = index_node(datum, sent_index)
    amrgrams = []
    if amr_index != -1:
        amrgrams[:] = get_amrgrams_node(datum, amr_index)
    else:
        amr_index = index_edge(datum, sent_index)
        if amr_index != -1:
            amrgrams[:] = get_amrgrams_edge(datum, amr_index)
        else:
            continue
    sentence = datum[POS]
    ngrams = get_ngrams(sentence, sent_index)
    amrgrams.extend(ngrams)
    test_examples.append(["0" for i in range(len(attributes))])
    for g in amrgrams:
        try:
            i = attributes.index(g)
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
    fout.write("@dataset amr-grams\n")
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
