from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from tabulate import tabulate
import pickle
from datetime import datetime

import constants

def plot_training_history(*histories, filename, patience):
    
    histories = [deepcopy(history) for history in histories]
    
    for history in histories:
        for k, v in history.items():
            history[k] = v[:-patience]
            
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    
    if all(['cl_loss' in history for history in histories]):
        cl_loss = []
        for history in histories:
            cl_loss.extend(history['cl_loss'])
    
    if all(['dist_loss' in history for history in histories]):
        dist_loss = []
        for history in histories:
            dist_loss.extend(history['dist_loss'])
            
    for history in histories:
        acc.extend(history['accuracy'])
        val_acc.extend(history['val_accuracy'])
        
        loss.extend(history['loss'])
        val_loss.extend(history['val_loss'])
    
    epochs_range = range(len(acc))
    line_list = []
    line = 0
    for history in histories:
        line = line + len(history['accuracy'])
        line_list.append(line)
    
    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label = 'Training Accuracy')
    plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
    for i, line in enumerate(line_list):
        plt.plot([line, line], [0, 1], color='k', linestyle='--', linewidth=1)
    plt.legend(loc = 'lower right')
    plt.ylim(0,1)
    plt.title('Training and Validation Accuracy', fontsize = 10)
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label = 'Training Loss')
    plt.plot(epochs_range, val_loss, label = 'Validation Loss')
    if 'dist_loss' in histories[0]:
        plt.plot(epochs_range, dist_loss, label = 'Distillation Loss')
        if 'cl_loss' in histories[0]:
            plt.plot(epochs_range, cl_loss, label = 'Classification Loss')
            plt.ylim(0, np.max([loss, val_loss, cl_loss, dist_loss]))
        else:
            plt.ylim(0, np.max([loss, val_loss, dist_loss]))
    else:
        plt.ylim(0, np.max([loss, val_loss]))
    for i, line in enumerate(line_list):
        plt.plot([line, line], [0, 5], color='k', linestyle='--', linewidth=1)
    plt.legend(loc = 'upper right')
    plt.title('Training and Validation Loss', fontsize = 10)
    plt.savefig(filename, bbox_inches = 'tight', dpi = 1000)
    try:
        get_ipython
        plt.show()
    except NameError:
        plt.clf()

def evaluate_model(model, test_ds, class_list, filename):
    
    test_preds = model.predict(test_ds)
    test_preds = np.argmax(test_preds, axis = -1)
    
    test_labels = np.concatenate([y for x, y in test_ds])
    test_labels = np.argmax(test_labels, axis = -1)
    
    cm = confusion_matrix(test_labels, test_preds, labels = np.arange(len(class_list)))
    
    cm_disp = ConfusionMatrixDisplay(cm, display_labels = class_list)
    _, ax = plt.subplots(figsize = (6,6))
    cm_disp.plot(ax = ax, xticks_rotation = 'vertical')
    plt.suptitle(f'Confusion Matrix', fontsize = 10)
    plt.savefig(filename, bbox_inches = 'tight', dpi = 1000)
    try:
        get_ipython
        plt.show()
    except NameError:
        plt.clf()
        
    accuracy = accuracy_score(test_labels, test_preds)
    
    print(f'\nAccuracy: {accuracy*100:.2f}%')
    
    report = classification_report(test_labels, test_preds, target_names = class_list, labels = np.arange(len(class_list)))
    print(report)
    
    return cm, accuracy, report

def get_basic_results(basic_results, basic_results_file, save=True):
    results_table = (
        tabulate(zip(range(10), basic_results['accuracy'], basic_results['n_epochs'], basic_results['train_time']), headers = ['Validation Fold', 'Validation Accuracy', 'Training Epochs', 'Training Time'])
        )
    print(results_table)
    if save:
        with open(basic_results_file, 'w') as f:
            f.write(results_table)

    basic_max_acc_idx = np.argmax(basic_results['accuracy'])
    basic_max_acc = np.max(basic_results['accuracy'])

    basic_min_acc_idx = np.argmin(basic_results['accuracy'])
    basic_min_acc = np.min(basic_results['accuracy'])

    basic_avg_acc = np.mean(basic_results['accuracy'])
    basic_std_acc = np.std(basic_results['accuracy'])
    basic_avg_train_time = np.mean(basic_results['train_time'])

    results_text = (
        f'\nMinimum accuracy for Basic FER model: {basic_min_acc:.02%} (Fold no. {basic_min_acc_idx})\n'
        f'Maximum accuracy for Basic FER model: {basic_max_acc:.02%} (Fold no. {basic_max_acc_idx})\n'
        f'Average accuracy for Basic FER model: {basic_avg_acc:.02%}\n'
        f'Standard deviation for Basic FER model accuracy: {basic_std_acc:.02%}\n'
        f'\nAverage training time for Basic FER model: {basic_avg_train_time:.02f} seconds'
        )
    print(results_text)
    if save:
        with open(basic_results_file, 'a') as f:
            f.write(results_text)

    best_val_fold = basic_max_acc_idx

    return best_val_fold

def get_cont_results(
        cont_results,
        basic_results,
        num_basic_classes,
        emotion_list,
        cont_images_dir,
        cont_fold,
        cont_val_fold,
        basic_emotion_list,
        complex_emotion_list,
        patience,
        cont_results_file,
        save = True
        ):
    cont_results = deepcopy(cont_results)
    cont_results['accuracy'].insert(0, basic_results['accuracy'][cont_val_fold])

    num_observed_labels = list(range(num_basic_classes, len(emotion_list)+1))

    plt.plot(num_observed_labels, cont_results['accuracy'])
    plt.xlabel('Number of observed labels')
    plt.ylabel('Accuracy')
    plt.ylim([0,1])
    plt.title('Continual Learning performance with each additional label')
    plt.savefig(cont_images_dir / 'cont_performance.jpg', bbox_inches = 'tight', dpi = 1000)
    try:
        get_ipython
        plt.show()
    except NameError:
        plt.clf()
    
    results_text = (
        f'----Continual Learning Results----'
        f'\nUsing complex emotion list: {cont_fold}'
        f'\nAverage Step Accuracy: {np.mean(cont_results["accuracy"]):.2%}\n'
    )
    print(results_text)
    if save:
        with open(cont_results_file, 'w') as f:
            f.write(results_text)

    cont_accuracy_table = []
    for i, accuracy in enumerate(cont_results['accuracy']):
        if i == 0:
            cont_accuracy_table.append(['Basic FER Phase', num_basic_classes,
                                        f'{num_basic_classes} Basic Emotions',
                                        f'{accuracy:.2%}', basic_results['n_epochs'][cont_val_fold]])
        else:
            cont_accuracy_table.append([f'Continual Learning Phase ({i})', num_basic_classes + i,
                                        complex_emotion_list[i-1],
                                        f'{accuracy:.2%}', cont_results['n_epochs'][i-1]])

    results_table = (
        tabulate(cont_accuracy_table, headers = ['Iteration', 'Number of labels', 'New Emotion Class', 'Accuracy', 'Training Epochs'])
        )
    print(results_table)
    if save:
        with open(cont_results_file, 'a') as f:
            f.write(results_table)

    plot_training_history(basic_results['history'][cont_val_fold],
                          *[i for j in zip(cont_results['history']) for i in j],
                          filename = cont_images_dir / 'cont_training_history_all.jpg',
                          patience = patience
                         )

    plt.figure(figsize = (20,40))
    for i, cm in enumerate(cont_results['cm']):
        cm_disp = ConfusionMatrixDisplay(cm, display_labels = basic_emotion_list + complex_emotion_list[:i+1])
        ax = plt.subplot(5,3,i+1)
        cm_disp.plot(ax = ax, xticks_rotation = 'vertical')
        plt.title(f'Iteration {i+1} | New emotion: {complex_emotion_list[i]}')
    plt.tight_layout(h_pad = 2, w_pad = 0)
    plt.savefig(cont_images_dir / 'cont_all_cms.jpg')
    try:
        get_ipython
        plt.show()
    except NameError:
        plt.clf()

def get_fewshot_results(fewshot_results, fewshot_results_file, complex_emotion_list, save = True):
    results_table = (
        tabulate(zip(complex_emotion_list, fewshot_results['accuracy'], fewshot_results['n_epochs'], fewshot_results['train_time']), headers = ['Expression', 'Validation Accuracy', 'Training Epochs', 'Training Time'])
        )
    print(results_table)
    if save:
        with open(fewshot_results_file, 'w') as f:
            f.write(results_table)

def get_overall_cont_results(num_basic_classes, cont_val_fold, excluded_emotions, save = True):
    if len(excluded_emotions) > 1:
        cont_results_file = constants.cont_results_dir / f'{datetime.now().strftime("%Y%m%d")}_overall_cont_results_exclude_{"_".join([x.lower() for x in excluded_emotions if x.lower() != "neutral"])}.txt'
        cont_image_file = constants.cont_images_dir / f'cont_step_accuracy_best_avg_worst_exclude_{"_".join([x.lower() for x in excluded_emotions if x.lower() != "neutral"])}.jpg'
    else:
        cont_results_file = constants.cont_results_dir / f'{datetime.now().strftime("%Y%m%d")}_overall_cont_results.txt'
        cont_image_file = constants.cont_images_dir / f'cont_step_accuracy_best_avg_worst.jpg'

    with open(constants.basic_results_dir / f'basic_results.pkl', 'rb') as f:
        basic_results = pickle.load(f)

    step_accuracy_list = []
    for cont_fold in range(10):
        if len(excluded_emotions) > 1:
            cont_results_dir = constants.cont_results_dir / f'cont_fold_{cont_fold}_exclude_{"_".join([x.lower() for x in excluded_emotions if x.lower() != "neutral"])}'
        else:
            cont_results_dir = constants.cont_results_dir / f'cont_fold_{cont_fold}'
        with open(cont_results_dir / f'cont_results.pkl', 'rb') as f:
            cont_results = pickle.load(f)
        cont_results = deepcopy(cont_results)
        cont_results['accuracy'].insert(0, basic_results['accuracy'][cont_val_fold])
        step_accuracy_list.append(cont_results['accuracy'])

    step_accuracy = np.array(step_accuracy_list)
    avg_step_accuracy = np.mean(step_accuracy, axis = 0)
    overall_accuracy = np.mean(avg_step_accuracy)

    results_text = (
        f'----Overall Continual Learning Results----'
        f'\nOverall Accuracy: {overall_accuracy:.4f}\n'
    )
    print(results_text)
    if save:
        with open(cont_results_file, 'w') as f:
            f.write(results_text)

    def closest_idx(array, K):
        array = np.asarray(array)
        idx = (np.abs(array - K)).argmin()
        return idx

    overall_accuracies = np.mean(step_accuracy, axis = 1)

    min_overall_accuracy_idx = np.argmin(overall_accuracies)
    max_overall_accuracy_idx = np.argmax(overall_accuracies)
    near_avg_overall_accuracy_idx = closest_idx(overall_accuracies, overall_accuracy)

    max_overall_accuracy = step_accuracy[max_overall_accuracy_idx]
    near_avg_overall_accuracy = step_accuracy[near_avg_overall_accuracy_idx]
    min_overall_accuracy = step_accuracy[min_overall_accuracy_idx]

    num_observed_labels = list(range(num_basic_classes, num_basic_classes+step_accuracy.shape[-1]))

    results_table = (
        tabulate(zip(range(16),
                 np.round(max_overall_accuracy, 4),
                 np.round(near_avg_overall_accuracy, 4),
                 np.round(min_overall_accuracy, 4)), headers = ['Step', 'Best Overall Accuracy', 'Near-Average Overall Accuracy', 'Worst Overall Accuracy']))
    print(results_table)
    if save:
        with open(cont_results_file, 'a') as f:
            f.write(results_table)

    step_accuracy_titles = ['Best Accuracy', 'Near-Average to CLA Accuracy', 'Worst Accuracy']

    plt.figure(figsize = (15,5))
    for i, accuracy in enumerate((max_overall_accuracy, near_avg_overall_accuracy, min_overall_accuracy)):
        plt.subplot(1, 3, i+1)
        plt.plot(num_observed_labels, accuracy)
        plt.xticks(list(range(num_basic_classes, num_basic_classes+step_accuracy.shape[-1])))
        plt.xlabel('Number of observed labels')
        plt.ylabel('Average Step Accuracy')
        plt.ylim([0,1])
        plt.title(step_accuracy_titles[i])
    #plt.suptitle(f'Continual Learning Step Accuracy over 10 Shuffled Complex Emotion Lists')
    plt.tight_layout()
    plt.savefig(cont_image_file, bbox_inches = 'tight', dpi = 1000)
    try:
        get_ipython
        plt.show()
    except NameError:
        plt.clf()