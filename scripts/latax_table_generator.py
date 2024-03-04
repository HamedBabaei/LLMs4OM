# -*- coding: utf-8 -*-
import pandas as pd
import os

rag_df = pd.read_csv("../experiments/results/rag-hybrid-model-results.csv")
ir_df = pd.read_csv("../experiments/results/retrieval-model-results.csv")
irs = ['AdaRetrieval', 'TFIDFRetrieval', 'SpecterBERTRetrieval', 'BERTRetrieval']

tables_path = '../experiments/tables'

ir_df = ir_df[ir_df['model'].isin(irs)]

print("size of RAG:", rag_df.shape)
print("size of Retriever:", ir_df.shape)


def retriever_to_latax(chosen_track, ir_df, rounds):
    model_mapper = {"AdaRetrieval": "Ada", "BERTRetrieval": "sentence-BERT",
                    "SpecterBERTRetrieval": "SPECTER2", "TFIDFRetrieval": "TFIDF"}
    encoder_mapper = {"label": "C", "label-children": "CC", "label-parent": "CP"}
    table = """
    \\begin{table}[h]
         \\centering
         \\caption{Retrieval models results --- \\textsc{TRACK} track -- Rep is the representation type.} \\label{tab:ir_LABEL}
         \\begin{tabular}{|l|c|l|r|r|r|r|r|r|r|r|r|}
             \\hline
             \\multirow{2}{*}{\\textbf{Model}}  & \\multirow{2}{*}{\\textbf{Rep}}  & \\multirow{2}{*}{\\textbf{Task}} &  \\multicolumn{3}{c|}{\\textbf{$Top_k=5$ Results}} &  \\multicolumn{3}{c|}{\\textbf{$Top_k=10$ Results}} &  \\multicolumn{3}{c|}{\\textbf{$Top_k=20$ Results}}\\\\
             \\cline{4-12}
              & & & Prec & Rec & F1& Prec & Rec & F1& Prec & Rec & F1 \\\\
             \\hline
    """.replace("TRACK", chosen_track).replace("LABEL", chosen_track)
    for encoder, encoder_df in ir_df.groupby("encoder-representation"):
        for track, track_df in encoder_df.groupby("track"):
            if track == chosen_track:
                track_df['precision'] = track_df['precision'].apply(lambda x: float(x)).round(rounds)
                track_df['recall'] = track_df['recall'].apply(lambda x: float(x)).round(rounds)
                track_df['f1-score'] = track_df['f1-score'].apply(lambda x: float(x)).round(rounds)
                for task, task_df in track_df.groupby("ontology-name"):
                    # print(task)
                    for model, model_df in task_df.groupby("model"):
                        # assert model_df.shape[0] == 3
                        result_dict = {}
                        for model, top_k, prec, rec, f1 in zip(model_df['model'], model_df['model-config'],
                                                               model_df['precision'],
                                                               model_df['recall'], model_df['f1-score']):
                            # print(f"{top_k} == {model}  & {prec} & {rec} & {f1} \\\\")
                            result_dict[top_k] = {"model": model, "p": prec, "r": rec, "f": f1}
                        table += f"\t{model_mapper.get(model)}  & {encoder_mapper.get(encoder)} & {task} "
                        # print(result_dict)
                        for tk in [5, 10, 20]:
                            table += f" & {result_dict[tk]['p']} & {result_dict[tk]['r']} & {result_dict[tk]['f']}"

                        table += "\\\\ \n"
                    table += "\t\\hline \n"
                    # print()
                # print()
        # print()
    table += """    \\end{tabular}
\\end{table}
        """
    return table


def rag_to_latax(chosen_track, rag_df, rounds=2):
    model_mapper = {"ChatGPTOpenAIAdaRAG": "GPT-3.5 + Ada",
                    "FalconAdaRAG": "Falcon-7B + Ada",
                    "FalconBertRAG": "Falcon-7B + BERT",
                    "LLaMA7BAdaRAG": "LLaMA-2-7B + Ada",
                    "LLaMA7BBertRAG": "LLaMA-2-7B + BERT",
                    "MPTAdaRAG": "MPT-7B + Ada",
                    "MPTBertRAG": "MPT-7B + BERT",
                    "MambaLLMBertRAG": "Mamba-2.8B + BERT",
                    "MambaLLMAdaRAG": "Mamba-2.8B + Ada",
                    "MistralAdaRAG": "Mistral-7B + Ada",
                    "MistralBertRAG": "Mistral-7B + BERT",
                    "VicunaAdaRAG": "Vicuna-7B + Ada",
                    "VicunaBertRAG": "Vicuna-7B + BERT"

                    }
    encoder_mapper = {"label": "C", "label-children": "CC", "label-parent": "CP"}

    Id = 1
    overall_table = ""
    for encoder, encoder_df in rag_df.groupby("encoder-representation"):
        table = """\\begin{table}
        \\centering
        \\small
        \\caption{LLM models results --- \\textsc{TRACKNAME} track -- Rep is the representation type. Retriever model Top-k is set to 5. PART NO } \\label{tab:llm_TAG}
        \\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|}
            \\hline
             \\multirow{2}{*}{\\textbf{Model}}  & \\multirow{2}{*}{\\textbf{Rep}}  & \\multirow{2}{*}{\\textbf{Task}} &  \\multicolumn{3}{c|}{\\textbf{Results}} \\\\
             \\cline{4-6}
              & & & Prec & Rec & F1  \\\\
            \\hline \n""".replace("TRACKNAME", chosen_track).replace("TAG", chosen_track + str(Id)).replace("NO",
                                                                                                            str(Id))

        for track, track_df in encoder_df.groupby("track"):
            if track == chosen_track:
                # print("TRACK:", track)
                track_df['precision'] = track_df['precision'].apply(lambda x: float(x)).round(rounds)
                track_df['recall'] = track_df['recall'].apply(lambda x: float(x)).round(rounds)
                track_df['f1-score'] = track_df['f1-score'].apply(lambda x: float(x)).round(rounds)
                for task, task_df in track_df.groupby("ontology-name"):
                    # print(task)
                    for model, model_df in task_df.groupby("model"):
                        result_dict = {}
                        for model, _, prec, rec, f1 in zip(model_df['model'], model_df['model-config'],
                                                           model_df['precision'],
                                                           model_df['recall'], model_df['f1-score']):
                            # print(f"{top_k} == {model}  & {prec} & {rec} & {f1} \\\\")
                            result_dict = {"model": model, "p": prec, "r": rec, "f": f1}
                        table += f"\t{model_mapper[model]}  & {encoder_mapper.get(encoder)} & {task} "
                        # print(result_dict)
                        # for tk in [5, 10, 20]:
                        table += f" &  {result_dict['p']} &  {result_dict['r']} & {result_dict['f']}  \\\\  \n"
                    table += "\t\\hline \n"
                    # print()
                    # break
                # print()
                # break
        table += """\\end{tabular}
    \\end{table}"""

        overall_table += table
        overall_table += "\n\n\n\n\n\n\n\n\n"
        Id = Id + 1
    return overall_table


if __name__ == "__main__":

    tracks = ['commonkg', 'bio-ml', 'anatomy', 'mse', 'biodiv', 'phenotype']
    for track in tracks:
        track_table_ir = retriever_to_latax(chosen_track=track, ir_df=ir_df, rounds=2)
        open(os.path.join(tables_path, track + "_retriever.tex"), "w").write(track_table_ir)
        track_table_rag = rag_to_latax(chosen_track=track, rag_df=rag_df, rounds=2)
        open(os.path.join(tables_path, track + "_rag.tex"), "w").write(track_table_rag)
