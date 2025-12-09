from rnn_attention import rnn_attention_caller
from transformer_att import transformer_attention_caller

if __name__ == '__main__':
	import sys
	import torch
	import pandas as pd

	model_path = sys.argv[1]
	model_type = sys.argv[2] # rnn/transformer
	data_path  = sys.argv[3]
	output_path = sys.argv[4]

	#no need to add prefix or suffix in the paths
	# weights = torch.load(model_path)

	DATA = pd.read_csv(data_path)
	if model_type == 'rnn':
		output_df = rnn_attention_caller(
			input_data = DATA,
			model_path = model_path,
			device='cuda' if torch.cuda.is_available() else 'cpu'
		)
	else:
		output_df = transformer_attention_caller(
			input_data = DATA,
			model_path = model_path,
			device='cuda' if torch.cuda.is_available() else 'cpu'
		)


	# no need to hardcode anything here as well
	output_df.to_csv(output_path ,index=False)
