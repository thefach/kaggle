from paremeters import PARAM 

def get_exploded_df(d):

    """ Explode the responses and prompts from a list of strings moving each prompt and its response in a row """

    d['prompt'] = d['prompt'].str.replace('null', 'None')
    d['prompt'] = d['prompt'].apply(eval)

    d['response_a'] = d['response_a'].str.replace('null', 'None')
    d['response_a'] = d['response_a'].apply(eval)

    d['response_b'] = d['response_b'].str.replace('null', 'None')
    d['response_b'] = d['response_b'].apply(eval)

    return d.explode(['prompt', 'response_a', 'response_b']).reset_index(drop=True)

def get_class_label(d):
    # Label conversion
    d["class_name"] = d[["winner_model_a", "winner_model_b" , "winner_tie"]].idxmax(axis=1)
    d["class_label"] = d.class_name.map(PARAM.name2label)

    return d

