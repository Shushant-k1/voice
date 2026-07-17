"""
Results Compiler.
Aggregates individual CSV benchmark outputs into a consolidated summary table.
"""

import os
import sys
import glob
import pandas as pd


def compile_markdown():
    """Load all result CSVs and print a consolidated metrics summary."""
    # Use the project root relative to this script's location
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    def norm_name(name):
        name = str(name).lower().replace('-', '_').strip()
        if name.endswith('_tts'):
            name = name[:-4]
        if name == 'indic_parler_tts':
            name = 'indic_parler'
        if name == 'openvoice_v2':
            name = 'openvoice'
        if name == 'f5_tts':
            name = 'f5'
        return name

    # 1. Load Latency & RTF
    all_rows = []
    for f in glob.glob('results/*_benchmark.csv'):
        df = pd.read_csv(f)
        df['model'] = df['model'].apply(norm_name)
        if 'latency_s' in df.columns and 'rtf' in df.columns:
            all_rows.append(df[['language', 'model', 'sentence_idx', 'latency_s', 'rtf']])
    
    if all_rows:
        df_all = pd.concat(all_rows).drop_duplicates(subset=['language', 'model', 'sentence_idx'])
        df_lat = df_all.groupby(['language', 'model'])[['latency_s', 'rtf']].mean().reset_index()
        df_lat = df_lat.rename(columns={'latency_s': 'latency'})
    else:
        df_lat = pd.DataFrame(columns=['language', 'model', 'latency', 'rtf'])

    # 2. Load Speaker Similarity
    df_sim = pd.DataFrame()
    if os.path.exists('results/speaker_similarity.csv'):
        df_sim = pd.read_csv('results/speaker_similarity.csv')
        df_sim['model'] = df_sim['model'].apply(norm_name)
        df_sim = df_sim.groupby(['language', 'model'])['cosine_similarity'].mean().reset_index()

    # 3. Load WER
    df_wer = pd.DataFrame()
    if os.path.exists('results/wer_results.csv'):
        df_wer = pd.read_csv('results/wer_results.csv')
        df_wer['model'] = df_wer['model'].apply(norm_name)
        df_wer = df_wer.groupby(['language', 'model'])['wer_normalized'].mean().reset_index()

    # 4. Load UTMOS
    df_utmos = pd.DataFrame()
    if os.path.exists('results/utmos_results.csv'):
        df_u = pd.read_csv('results/utmos_results.csv')
        langs = []
        models = []
        for path in df_u['audio_path']:
            parts = path.split('/')
            if len(parts) >= 3:
                langs.append(parts[1])
                fname = parts[2].replace('.wav', '')
                mname = fname.rsplit('_', 1)[0]
                models.append(mname)
            else:
                langs.append('unknown')
                models.append('unknown')
        df_u['language'] = langs
        df_u['model'] = models
        df_u['model'] = df_u['model'].apply(norm_name)
        df_utmos = df_u.groupby(['language', 'model'])['utmos_score'].mean().reset_index()

    # 5. Load Human MOS
    df_hmos = pd.DataFrame()
    if os.path.exists('results/human_mos.csv'):
        df_h = pd.read_csv('results/human_mos.csv')
        df_h['model'] = df_h['model'].apply(norm_name)
        df_hmos = df_h.groupby(['language', 'model'])['human_mos'].mean().reset_index()

    # Merge all metrics
    merged = df_lat
    if not df_sim.empty:
        merged = pd.merge(merged, df_sim, on=['language', 'model'], how='outer')
    if not df_wer.empty:
        merged = pd.merge(merged, df_wer, on=['language', 'model'], how='outer')
    if not df_utmos.empty:
        merged = pd.merge(merged, df_utmos, on=['language', 'model'], how='outer')
    if not df_hmos.empty:
        merged = pd.merge(merged, df_hmos, on=['language', 'model'], how='outer')

    merged = merged.sort_values(['language', 'model'])

    # Print summary table
    print("=== SUMMARY METRICS ===")
    print(f"{'Language':<10} {'Model':<20} {'Latency(s)':<12} {'RTF':<8} {'Similarity':<12} {'WER':<10} {'UTMOS':<8} {'Human MOS':<10}")
    print("-" * 90)
    for _, row in merged.iterrows():
        sim_val = f"{row['cosine_similarity']:.3f}" if 'cosine_similarity' in row and not pd.isna(row.get('cosine_similarity')) else "N/A"
        wer_val = f"{row['wer_normalized']*100:.2f}%" if 'wer_normalized' in row and not pd.isna(row.get('wer_normalized')) else "N/A"
        utmos_val = f"{row['utmos_score']:.2f}" if 'utmos_score' in row and not pd.isna(row.get('utmos_score')) else "N/A"
        hmos_val = f"{row['human_mos']:.2f}" if 'human_mos' in row and not pd.isna(row.get('human_mos')) else "N/A"
        print(f"| {row['language'].capitalize():<9} | {row['model'].upper():<18} | {row['latency']:.3f}s | {row['rtf']:.3f} | {sim_val:<10} | {wer_val:<8} | {utmos_val:<6} | {hmos_val:<9} |")


if __name__ == '__main__':
    compile_markdown()
