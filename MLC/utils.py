from collections import defaultdict
import codecs 

def merge_files(entities, output_folder):
    my_dict = defaultdict(list)
    output_file = codecs.open('predictions.tsv', 'w', 'utf-8')
    for i, entity in enumerate(entities):
        predictions = codecs.open(f'{output_folder}/{entity}/test.tsv', 'r', 'utf-8').read()
        for j, line in enumerate(predictions.splitlines()):
            if line == '':
                my_dict[j].append('EOS')
                continue
            data = line.split()
            token = data[0]
            prediction = data[2]
            if i == 0:
                my_dict[j].append(token)
                my_dict[j].append(prediction)
            else:
                my_dict[j].append(prediction)

    for k, v in my_dict.items():
        if v[0] == 'EOS':
            output_file.write("\n")
        else:
            if list(set(v[1:])) == ['O']:
                output_file.write(f"{v[0]} O\n")
            else: 
                new_array = v[1:]
                new_array = list(filter(lambda a: a != 'O', new_array))
                output_file.write(f"{v[0]} {' '.join(new_array)}\n")
    output_file.close()

def show_results(entities, output_folder):
    tp = 0
    fn = 0
    fp = 0
    print('\n')
    print('----------------------------- Scores per entity type --------------------------- \n')
    for i, entity in enumerate(entities):
        log_file = codecs.open(f'{output_folder}/{entity}/training.log', 'r', 'utf-8').read()
        for line in log_file.splitlines():
            if "f1-score" in line: 
                print(line)
                data = line.split()
                tp += int(data[2])
                fp += int(data[5])
                fn += int(data[8])

    print('\n')
    print('----------------------------- Overral scores  --------------------------- \n')
    micro_precision = tp/(tp+fp)
    micro_recall = tp/(tp+fn)
    micro_f1 = (2*micro_precision*micro_recall)/(micro_precision+micro_recall)
    print(f'Micro precision: {round(micro_precision*100, 2)}')
    print(f'Micro recall: {round(micro_recall*100, 2)}')
    print(f'Micro f1-score: {round(micro_f1*100, 2)}')