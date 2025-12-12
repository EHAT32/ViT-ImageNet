config = {
    #model size
    'img_size' : 64,
    'patch_size' : 16,
    'channel_num' : 3,
    'embedding_dim' : 192,
    'hidden_dim' : 192,
    'ffn_dim' : 192 * 4,
    'attention_head_number' : 3,
    'block_number' : 7,
    'class_number' : 200,
    
    #dropout probs
    'embedding_dropout' : 0.1,
    'ffn_dropoput' : 0.1,
    'multihead_attention_dropout' : 0.1,
    'attention_dropout' : 0.1,
    
    #optimizer
    'optimizer' : {
        'lr' : 1e-3,
        'betas' : (0.9, 0.95),
        'weight_decay' : 0.1,
        'eta_min' : 1e-6
    }
}