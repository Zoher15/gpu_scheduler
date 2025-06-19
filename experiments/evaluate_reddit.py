            second_model_kwargs = model_kwargs.copy()
            if 'num_return_sequences' in second_model_kwargs:
                del second_model_kwargs['num_return_sequences']
            if 'top_k_beams' in second_model_kwargs:
                del second_model_kwargs['top_k_beams']
            second_model_kwargs['do_sample'] = False
            
            second_responses = get_first_responses(model, processor_or_tokenizer, second_model_kwargs, second_prompts, [None] * len(second_prompts)) 