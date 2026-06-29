# Regras de Engenharia — Paradoxo Experiment

## Scripts de Experimento (GPU)

Todo script que roda treinamento em GPU deve seguir obrigatoriamente:

1. **Save incremental:** salvar resultados em JSON depois de CADA geração, não no final
2. **Resume automático:** ao iniciar, detectar progresso existente e continuar de onde parou
3. **Sintéticos em disco:** salvar dados sintéticos gerados em arquivo separado por geração, não manter em memória entre iterações
4. **Liberar memória antes de gerar:** deletar trainer, optimizer e train_ds ANTES de chamar generate_synthetic. Chamar gc.collect() e torch.cuda.empty_cache()
5. **Skip automático:** se o resultado final já existe completo, skip sem reprocessar

### Padrão de código:

```python
# Save incremental
gen_results.append(result)
json.dump(gen_results, open(result_path, "w"), indent=2)

# Resume
gen_results = json.load(open(result_path)) if result_path.exists() else []
start_gen = len(gen_results) + 1

# Free before generation
del trainer, train_ds, initial_state
gc.collect(); torch.cuda.empty_cache()

# Synthetic to disk
json.dump(synthetic, open(output_dir / f"syn_{key}_gen{gen}.json", "w"))
del synthetic; gc.collect(); torch.cuda.empty_cache()
```

## Batch Size

- RTX 3070 (8GB local): batch=2, grad_accum=8
- RTX 4000 Ada (20GB Athena): batch=4, grad_accum=4
- Effective batch deve ser sempre 16 (2 epochs × 2000 samples = 250 steps)
- Para FFT bf16: usar gradient_checkpointing=True obrigatoriamente


## Escrita Acadêmica

1. **Nunca usar traços como separadores ou pontuação:** substituir em-dashes (---), en-dashes (--) e traços (-) usados como pontuação por vírgulas, pontos, ou reestruturação da frase. Exceção: hífens em palavras compostas (e.g., "dose-response", "cross-backbone").
