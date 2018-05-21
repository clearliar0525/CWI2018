from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score

def execute_demo(language,algor):
    data = Dataset(language)

    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.testset)))


    baseline = Baseline(language,algor)
    
    freqdict1=baseline.freqdict(data.trainset+data.testset)
    
    posindex1=baseline.posdict(data.trainset+data.testset)

    baseline.train(data.trainset,freqdict1,posindex1)

    predictions = baseline.test(data.testset,freqdict1,posindex1)

    gold_labels = [sent['gold_label'] for sent in data.testset]

    report_score(gold_labels, predictions)


if __name__ == '__main__':
    execute_demo('english',"RandomForest")
    #execute_demo('english',"svm")
    #execute_demo('english',"tree")
    #execute_demo('english','LogisticRegression')
    
    #execute_demo('spanish',"RandomForest")
    #execute_demo('spanish',"svm")
    #execute_demo('spanish',"tree")
    #execute_demo('spanish','LogisticRegression')
    
