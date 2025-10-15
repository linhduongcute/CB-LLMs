import concepts

example_name = {'SetFit/sst2': 'text', 'ag_news': 'text', 'yelp_polarity': 'text', 'dbpedia_14': 'content', "Duyacquy/Pubmed-20k": 'abstract_text'}
concepts_from_labels = {"Duyacquy/Pubmed-20k": ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"],'SetFit/sst2': ["negative","positive"], 'yelp_polarity': ["negative","positive"], 'ag_news': ["world", "sports", "business", "technology"], 'dbpedia_14': ["company","education","artist","athlete","office","transportation","building","natural","village","animal","plant","album","film","written"]}
class_num = {"Duyacquy/Pubmed-20k": 5, 'SetFit/sst2': 2, 'ag_news': 4, 'yelp_polarity': 2, 'dbpedia_14': 14}

# Config for Roberta-Base baseline
finetune_epoch = { "Duyacquy/Pubmed-20k": 2, 'SetFit/sst2': 3, 'ag_news': 2, 'yelp_polarity': 2, 'dbpedia_14': 2}
finetune_mlp_epoch = {"Duyacquy/Pubmed-20k": 5, 'SetFit/sst2': 30, 'ag_news': 5, 'yelp_polarity': 3, 'dbpedia_14': 3}

# Config for CBM training
concept_set = {"Duyacquy/Pubmed-20k": concepts.pubmed,'SetFit/sst2': concepts.sst2, 'yelp_polarity': concepts.yelpp, 'ag_news': concepts.agnews, 'dbpedia_14': concepts.dbpedia}
#cbl_epochs = {'SetFit/sst2': 30, 'ag_news': 5, 'yelp_polarity': 3, 'dbpedia_14': 3}

cbl_epochs = { "Duyacquy/Pubmed-20k": 5,'SetFit/sst2': 10, 'ag_news': 3, 'yelp_polarity': 2, 'dbpedia_14': 2}
