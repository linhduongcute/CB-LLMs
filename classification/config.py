import concepts

example_name = {'SetFit/sst2': 'text', 'ag_news': 'text', 'yelp_polarity': 'text', 'dbpedia_14': 'content', 'Duyacquy/UCI_drug': 'review', 'Duyacquy/Ecommerce_text': 'text'}
concepts_from_labels = {'SetFit/sst2': ["negative","positive"], 'yelp_polarity': ["negative","positive"], 'ag_news': ["world", "sports", "business", "technology"], 'dbpedia_14': ["company","education","artist","athlete","office","transportation","building","natural","village","animal","plant","album","film","written"], 'Duyacquy/UCI_drug': ["rating 1", "rating 5", "rating 10"], 'Duyacquy/Ecommerce_text': ["Household", "Electronics", "Clothing & Accessories", "Books"]}
class_num = {'SetFit/sst2': 2, 'ag_news': 4, 'yelp_polarity': 2, 'dbpedia_14': 14, 'Duyacquy/UCI_drug': 3, 'Duyacquy/Ecommerce_text': 4}

# Config for Roberta-Base baseline
finetune_epoch = {'SetFit/sst2': 3, 'ag_news': 2, 'yelp_polarity': 2, 'dbpedia_14': 2, 'Duyacquy/UCI_drug': 2, 'Duyacquy/Ecommerce_text': 2}
finetune_mlp_epoch = {'SetFit/sst2': 30, 'ag_news': 5, 'yelp_polarity': 3, 'dbpedia_14': 3, 'Duyacquy/UCI_drug': 3, 'Duyacquy/Ecommerce_text': 3}

# Config for CBM training
concept_set = {'SetFit/sst2': concepts.sst2, 'yelp_polarity': concepts.yelpp, 'ag_news': concepts.agnews, 'dbpedia_14': concepts.dbpedia, 'Duyacquy/UCI_drug': concepts.uci_drug, 'Duyacquy/Ecommerce_text': concepts.ecommerce_text}
#cbl_epochs = {'SetFit/sst2': 30, 'ag_news': 5, 'yelp_polarity': 3, 'dbpedia_14': 3}
cbl_epochs = {'SetFit/sst2': 10, 'ag_news': 3, 'yelp_polarity': 2, 'dbpedia_14': 2, 'Duyacquy/UCI_drug': 2, 'Duyacquy/Ecommerce_text': 2}
