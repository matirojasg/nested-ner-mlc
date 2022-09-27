# coding: utf-8

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
import ast
from collections import defaultdict
from dataset import End2EndDataset
from utils.torch_util import calc_f1
from utils.path_util import from_project_root

# Function used to obtain complete nestings (internal and external entities).
def get_nestings(entities):
  nestings = [] 
  total = []

  for e1 in entities:
    is_outer = True 
    possible_nested_entity = [e1]
    
    for e2 in entities:
      if e1!=e2:
        s_e1 = e1[1]
        e_e1 = e1[2]
        s_e2 = e2[1]
        e_e2 = e2[2]
        if ((s_e1>s_e2 and e_e1<e_e2) or (s_e1==s_e2 and e_e1<e_e2) or (s_e1>s_e2 and e_e1==e_e2)):
          is_outer = False 
        if (s_e2>=s_e1 and e_e2<=e_e1):
          if e1 not in total:
            total.append(e1)
          if e2 not in total:
            total.append(e2)
          possible_nested_entity.append(e2)
    
    if len(possible_nested_entity)==1:
      is_outer = False
    
    if is_outer:
      possible_nested_entity.sort(key=lambda x: (x[2]-x[1], x[0]), reverse=True)
      if possible_nested_entity not in nestings:
        nestings.append(possible_nested_entity)
  return nestings, total

def metric(pred, gold):
  tp = 0
  fn = 0
  fp = 0
  support = 0
  for p, g in zip(pred, gold):
    for entity in p: 
      if entity in g: 
        tp+=1
      if entity not in g:
        fp+=1

    for entity in g:
      support+=1
      if entity not in p:
        fn+=1
  
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1 = (2*precision*recall)/(precision+recall)

  return precision, recall, f1, support

def nesting_metric(pred_labels, true_labels):
  
  nesting_tp = 0
  nesting_fn = 0
  nesting_fp = 0
  support = 0
  for sent_pred_labels, sent_test_labels in zip(pred_labels, true_labels):
    pred_nestings, tp = get_nestings(sent_pred_labels)
    test_nestings, tt = get_nestings(sent_test_labels)
    
    for nesting in test_nestings:
      support+=1
      if nesting in pred_nestings:
        nesting_tp+=1
      else:
        nesting_fn+=1

    for nesting in pred_nestings:
      if nesting not in test_nestings:
        nesting_fp+=1
  nesting_precision = nesting_tp/(nesting_tp+nesting_fp)
  nesting_recall = nesting_tp/(nesting_tp+nesting_fn)
  nesting_f1 = 2*(nesting_precision*nesting_recall)/(nesting_precision+nesting_recall)
  return nesting_precision, nesting_recall, nesting_f1, support

def nested_metric(pred_labels, true_labels):
  
  nested_tp = 0
  nested_fn = 0
  nested_fp = 0
  support = 0

  for sent_pred_labels, sent_test_labels in zip(pred_labels, true_labels):
    pred_nestings, tp = get_nestings(sent_pred_labels)
    test_nestings, tt = get_nestings(sent_test_labels)
    
    for nesting in test_nestings:
      for entity in nesting:
        support+=1
        if entity in sent_pred_labels:
          nested_tp+=1
        else:
          nested_fn+=1

    for nesting in pred_nestings:
      for entity in nesting:
        if entity not in sent_test_labels:
          nested_fp+=1
    
  nested_precision = nested_tp/(nested_tp+nested_fp)
  nested_recall = nested_tp/(nested_tp+nested_fn)
  nested_f1 = 2*(nested_precision*nested_recall)/(nested_precision+nested_recall)
  return nested_precision, nested_recall, nested_f1, support

def inner_metric(pred_labels, true_labels):
  support = 0
  inner_tp = 0
  inner_fn = 0
  inner_fp = 0

  for sent_pred_labels, sent_test_labels in zip(pred_labels, true_labels):
    pred_nestings, tp = get_nestings(sent_pred_labels)
    test_nestings, tt = get_nestings(sent_test_labels)

    for nesting in test_nestings:
      for entity in nesting[1:]:
        support+=1
        if entity in sent_pred_labels:
          inner_tp+=1
        else:
          inner_fn+=1

    for nesting in pred_nestings:
      for entity in nesting[1:]:
        if entity not in sent_test_labels:
          inner_fp+=1

  inner_precision = inner_tp/(inner_tp+inner_fp)
  inner_recall = inner_tp/(inner_tp+inner_fn)
  inner_f1 = 2*(inner_precision*inner_recall)/(inner_precision+inner_recall)
  return inner_precision, inner_recall, inner_f1, support

def outer_metric(pred_labels, true_labels):
  
  outer_tp = 0
  outer_fn = 0
  outer_fp = 0
  support = 0
  for sent_pred_labels, sent_test_labels in zip(pred_labels, true_labels):
    pred_nestings, tp = get_nestings(sent_pred_labels)
    test_nestings, tt = get_nestings(sent_test_labels)
    
    for nesting in test_nestings:
      support+=1
      if nesting[0] in sent_pred_labels:
        outer_tp+=1
      else:
        outer_fn+=1

    for nesting in pred_nestings:
      if nesting[0] not in sent_test_labels:
        outer_fp+=1
  
  outer_precision = outer_tp/(outer_tp+outer_fp)
  outer_recall = outer_tp/(outer_tp+outer_fn)
  outer_f1 = 2*(outer_precision*outer_recall)/(outer_precision+outer_recall)
  return outer_precision, outer_recall, outer_f1, support 

def flat_metric(pred_labels, true_labels):
  
  flat_tp = 0
  flat_fn = 0
  flat_fp = 0
  support = 0
  total = 0
  for sent_pred_labels, sent_test_labels in zip(pred_labels, true_labels):
    pred_nestings, tp = get_nestings(sent_pred_labels)
    pred_flat_entities = []
    for entity in sent_pred_labels:
      is_nested = False
      for nesting in pred_nestings:
        if entity in nesting:
          is_nested = True
      if not is_nested:
        pred_flat_entities.append(entity)
    

    test_nestings, tt = get_nestings(sent_test_labels)
    test_flat_entities = []
    for entity in sent_test_labels:
   
      is_nested = False
      for nesting in test_nestings:
        if entity in nesting:
          is_nested = True
      if not is_nested:
        test_flat_entities.append(entity)

    

    for entity in test_flat_entities:
      support+=1
      if entity in sent_pred_labels:
        flat_tp+=1
      else:
        flat_fn+=1

    for entity in pred_flat_entities:
      if entity not in sent_test_labels:
        flat_fp+=1

  flat_precision = flat_tp/(flat_tp+flat_fp)
  flat_recall = flat_tp/(flat_tp+flat_fn)
  flat_f1 = 2*(flat_precision*flat_recall)/(flat_precision+flat_recall)
  return flat_precision, flat_recall, flat_f1, support 

def multilabel_metric(pred_labels, true_labels):
  multilabel_tp = 0
  multilabel_fn = 0
  multilabel_fp = 0
  support = 0
  for sent_pred_labels, sent_test_labels in zip(pred_labels, true_labels):
    pred_nestings, tp = get_nestings(sent_pred_labels)
    test_nestings, tt = get_nestings(sent_test_labels)
   
    test_multilabel_entities = defaultdict(list)
    for nesting in test_nestings:
      for entity in nesting:
        test_multilabel_entities[(entity[1], entity[2])].append(entity[0])
    
    for k, v in test_multilabel_entities.items():
      if len(v)>1:
        support+=1
        all_predicted = True
        for entity in v:
          if [entity, k[0], k[1]] not in sent_pred_labels:
            all_predicted = False

        if all_predicted:
          multilabel_tp+=1
        else:
          multilabel_fn+=1

      
    pred_multilabel_entities = defaultdict(list)
    for nesting in pred_nestings:
      for entity in nesting:
        pred_multilabel_entities[(entity[1], entity[2])].append(entity[0])
       
    for k, v in pred_multilabel_entities.items():
      if len(v)>1:
        all_predicted = True
        for entity in v:
          if [entity, k[0], k[1]] not in sent_test_labels:
            all_predicted = False

        if not all_predicted:
          multilabel_fp+=1
          
  multilabel_precision = multilabel_tp/(multilabel_tp+multilabel_fp) if multilabel_tp+multilabel_fp!=0 else 0
  multilabel_recall = multilabel_tp/(multilabel_tp+multilabel_fn) if multilabel_tp+multilabel_fn!=0 else 0
  multilabel_f1 = 2*(multilabel_precision*multilabel_recall)/(multilabel_precision+multilabel_recall) if multilabel_precision+multilabel_recall!=0 else 0
  return multilabel_precision, multilabel_recall, multilabel_f1, support

def same_nesting_type_metric(pred_labels, true_labels):
  snt_tp = 0
  snt_fn = 0
  snt_fp = 0
  support = 0

  for sent_pred_labels, sent_test_labels in zip(pred_labels, true_labels):
    pred_nestings, tp = get_nestings(sent_pred_labels)
    test_nestings, tt = get_nestings(sent_test_labels)

    snt_test = []
    for nesting in test_nestings:
      outer = nesting[0]
      stn = [outer]
      for inner in nesting[1:]:

        if inner[0]==outer[0]:
          stn.append(inner)
  
      if len(stn)>1: snt_test.append(stn)

    snt_pred = []
    for nesting in pred_nestings:
      outer = nesting[0]
      stn = [outer]
      for inner in nesting[1:]:
        if inner[0]==outer[0]:
          stn.append(inner)
          
      if len(stn)>1: snt_pred.append(stn)
        

    
    for nesting in snt_test:
      all_equal = True
      for entity in nesting:
        if entity not in sent_pred_labels:
          all_equal = False

      if all_equal:
        snt_tp+=1
      else:
        snt_fn+=1

    for nesting in snt_pred:
      all_equal = True
      for entity in nesting:
        if entity not in sent_test_labels:
          all_equal = False
      
      if not all_equal:
        snt_fp+=1

    support+=len(snt_test)

  
  snt_precision = snt_tp/(snt_tp+snt_fp) if snt_tp+snt_fp!=0 else 0
  snt_recall = snt_tp/(snt_tp+snt_fn) if snt_tp+snt_fn!=0 else 0
  snt_f1 = 2*(snt_precision*snt_recall)/(snt_precision+snt_recall) if snt_precision+snt_recall!=0 else 0
  return snt_precision, snt_recall, snt_f1, support

def is_multilabel_entity(nesting):
  for entity in nesting:
    if entity[1]!=nesting[0][1] or entity[2]!=nesting[0][2]:
      return False
  return True

def different_nesting_type_metric(pred_labels, true_labels):

  # Métrica multilabel
  dnt_tp = 0
  dnt_fn = 0
  dnt_fp = 0
  support = 0

  for sent_pred_labels, sent_test_labels in zip(pred_labels, true_labels):

    # Obtenemos todas las anidaciones
    pred_nestings, tp = get_nestings(sent_pred_labels)
    test_nestings, tt = get_nestings(sent_test_labels)
    
    dnt_test = []
    for nesting in test_nestings:
      if not is_multilabel_entity(nesting):
        outer = nesting[0]
        dtn = [outer]
        for inner in nesting[1:]:
          if inner[0]!=outer[0]:
            dtn.append(inner)

        if len(dtn)>1: dnt_test.append(dtn)
    
    dnt_pred = []
    for nesting in pred_nestings:
      if not is_multilabel_entity(nesting):
        outer = nesting[0]
        dtn = [outer]
        for inner in nesting[1:]:
          if inner[0]!=outer[0]:
            dtn.append(inner)

        if len(dtn)>1: dnt_pred.append(dtn)
    

   

    for nesting in dnt_test:
      all_equal = True
      for entity in nesting:
        if entity not in sent_pred_labels:
          all_equal = False
      
      if all_equal:
        dnt_tp+=1
      else:
        dnt_fn+=1
    for nesting in dnt_pred:
      all_equal = True
      for entity in nesting:
        if entity not in sent_test_labels:
          all_equal = False
      
      if not all_equal:
        dnt_fp+=1

    support+=len(dnt_test)

  
  dnt_precision = dnt_tp/(dnt_tp+dnt_fp) if dnt_tp+dnt_fp!=0 else 0
  dnt_recall = dnt_tp/(dnt_tp+dnt_fn) if dnt_tp+dnt_fn!=0 else 0
  dnt_f1 = 2*(dnt_precision*dnt_recall)/(dnt_precision+dnt_recall) if dnt_precision+dnt_recall!=0 else 0
  return dnt_precision, dnt_recall, dnt_f1, support

def evaluate_e2e(model, data_url, bsl_model=None):
    """ evaluating end2end model on dataurl
    Args:
        model: trained end2end model
        data_url: url to test dataset for evaluating
        bsl_model: trained binary sequence labeling model
    Returns:
        ret: dict of precision, recall, and f1
    """
    print("\nevaluating model on:", data_url, "\n")
    dataset = End2EndDataset(data_url, next(model.parameters()).device, evaluating=True)
    loader = DataLoader(dataset, batch_size=200, collate_fn=dataset.collate_func)
    ret = {'precision': 0, 'recall': 0, 'f1': 0}

    # switch to eval mode
    model.eval()
    golden_entities = []
    boundary_entities = []
    with torch.no_grad():
        sentence_true_list, sentence_pred_list = list(), list()
        region_true_list, region_pred_list = list(), list()
        region_true_count, region_pred_count = 0, 0
        for data, sentence_labels, region_labels, records_list in loader:
            if bsl_model:
                pred_sentence_labels = torch.argmax(bsl_model.forward(*data), dim=1)
                pred_region_output, _ = model.forward(*data, pred_sentence_labels)
            else:
                try:
                    pred_region_output, pred_sentence_output = model.forward(*data)
                    # pred_sentence_output (batch_size, n_classes, lengths[0])
                    pred_sentence_labels = torch.argmax(pred_sentence_output, dim=1)
                    # pred_sentence_labels (batch_size, max_len)
                except RuntimeError:
                    print("all 0 tags, no evaluating this epoch")
                    continue

            # pred_region_output (n_regions, n_tags)
            pred_region_labels = torch.argmax(pred_region_output, dim=1).view(-1).cpu()
            # (n_regions)

            lengths = data[1]
            ind = 0
            for sent_labels, length, true_records in zip(pred_sentence_labels, lengths, records_list):
                pred_records = dict()
                for start in range(0, length):
                    if sent_labels[start] == 1:
                        if pred_region_labels[ind]>0: 
                            pred_records[(start,start+1)] = pred_region_labels[ind]
                        ind += 1
                        for end in range(start + 1, length):
                            if sent_labels[end] == 2:
                                if pred_region_labels[ind]:
                                    pred_records[(start,end+1)] = pred_region_labels[ind]
                                ind += 1

                entities = []
                for k, v in true_records.items():
                  entities.append([k[0], k[1], dataset.label_ids[v]])
                golden_entities.append(entities)
                
                entities = []
                
                for k, v in pred_records.items():
                    entities.append([k[0], k[1], v.item()])
                boundary_entities.append(entities)

                for region in true_records:
                    true_label = dataset.label_ids[true_records[region]]
                    pred_label = pred_records[region] if region in pred_records else 0
                    region_true_list.append(true_label)
                    region_pred_list.append(pred_label)
                for region in pred_records:
                    if region not in true_records:
                        region_pred_list.append(pred_records[region])
                        region_true_list.append(0)

            region_labels = region_labels.view(-1).cpu()
            region_true_count += int((region_labels > 0).sum())
            region_pred_count += int((pred_region_labels > 0).sum())

            pred_sentence_labels = pred_sentence_labels.view(-1).cpu()
            sentence_labels = sentence_labels.view(-1).cpu()
            for tv, pv, in zip(sentence_labels, pred_sentence_labels):
                sentence_true_list.append(tv)
                sentence_pred_list.append(pv)

        print("sentence head and tail labeling result:")
        print(classification_report(sentence_true_list, sentence_pred_list,
                                    target_names=['out-entity', 'head-entity', 'tail-entity','in-entity'], digits=6))

        print("region classification result:")
        print(classification_report(region_true_list, region_pred_list,
                                    target_names=list(dataset.label_ids)[:13], digits=6))
        
        p, r, f1, support = metric(boundary_entities, golden_entities)
        print(f'Boundary f1-score: {np.round(f1*100,1)}, support: {support}')

        _, _, flat_f1, support = flat_metric(boundary_entities, golden_entities)
        print(f'Boundary Flat f1-score: {np.round(flat_f1*100,1)}, support: {support}')

        _, _, snt_f1, support = same_nesting_type_metric(boundary_entities, golden_entities)
        print(f'BoundaryM SNT f1-score: {np.round(snt_f1*100,1)}, support: {support}')

        _, _, dnt_f1, support = different_nesting_type_metric(boundary_entities, golden_entities)
        print(f'Boundary DNT f1-score: {np.round(dnt_f1*100,1)}, support: {support}')

        _, _, nesting_f1, support = nesting_metric(boundary_entities, golden_entities)
        print(f'Boundary Nesting f1-score: {np.round(nesting_f1*100,1)}, support: {support}')

        _, _, nested_f1, support = nested_metric(boundary_entities, golden_entities)
        print(f'Boundary Nested f1-score: {np.round(nested_f1*100,1)}, support: {support}')

        _, _, inner_f1, support = inner_metric(boundary_entities, golden_entities)
        print(f'Boundary Inner f1-score: {np.round(inner_f1*100,1)}, support: {support}')

        _, _, outer_f1, support = outer_metric(boundary_entities, golden_entities)
        print(f'Boundary Outer f1-score: {np.round(outer_f1*100,1)}, support: {support}')
        print()
        
        ret = dict()
        tp = 0
        for pv, tv in zip(region_pred_list, region_true_list):
            if pv == tv == 0:
                continue
            if pv == tv:
                tp += 1
        fp = region_pred_count - tp
        fn = region_true_count - tp
        ret['precision'], ret['recall'], ret['f1'] = calc_f1(tp, fp, fn)

    return ret


def main():
    model_url = from_project_root("data/model/end2end_model_epoch2_0.715725.pt")
    test_url = from_project_root("data/germ/germ.test.iob2")
    model = torch.load(model_url)
    evaluate_e2e(model, test_url)
    pass


if __name__ == '__main__':
    main()