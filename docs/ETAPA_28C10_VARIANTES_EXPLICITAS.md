# Etapa 28C.10 — Variantes explícitas e níveis de evidência

**Base:** `feature/mass-properties`, commit `f8323c8e` (28C.8). **Objetivo:** duas montagens com rótulos obrigatórios e testes independentes, sem eleger padrão de estabilidade física.

## Escopo e contrato

- `openplaning/stability/source_variants.py` cria duas funções explícitas:
  - `assemble_lewandowski_2004_tabulated(hull, roll, rigid, roll_state=...)`: reproduz **Y_phi, K_phi, N_phi das Tabelas 6.2 e 6.3**, usando a geometria e a hipótese FSD informadas. `EvidenceLevel.DOCUMENTAL_FIDELITY` **não significa validade física**.
  - `assemble_lewandowski_1996_full_projection(hull, roll, rigid, roll_state=...)`: reproduz exatamente a *hipótese 28C.5*, combinando a projeção FPU de forças por bordo de 1996 com as derivadas de velocidades de 2004 e transformando a coluna restauradora para FSD. `EvidenceLevel.DIAGNOSTIC`.
- O nome da segunda interface identifica a origem da **coluna restauradora**, não uma reconstrução independente do modelo completo de 1996.
- A interface pública `openplaning.stability.from_resolved_planing_boat` foi **intencionalmente substituída** por um roteador com parâmetro obrigatório `formulation: SourceFormulation`. Invocá-la sem esse parâmetro lança `TypeError`.
- Para regressão histórica, a função original `openplaning.stability.transverse.from_resolved_planing_boat` **continua intacta**, assim como `from_resolved_planing_boat_fsd` e os assemblers anteriores. Isso protege os testes legados, mas quem importava o facade diretamente da raiz `openplaning.stability` deverá atualizar sua chamada.
- Não há seleção automática pelo tempo da publicação, pela classificação estável ou pela posição na enumeração.
- `SourceResolvedSystem` transporta `SourceFormulation` e `EvidenceLevel` tipados, dados de referência/quadros/unidades e limitações. O sistema conserva a matriz e os autovalores originais na propriedade `linear_system`.

## Coluna restauradora

### Tabulada 2004 — FSD, CG

Dados `audit_roll_sources`, avaliado diretamente das Tabelas 6.2 e 6.3:

\[
Y_\phi^{04}=2F_\phi^{04}\sin\beta,\qquad
N_\phi^{04}=Y_\phi^{04}(L_{CP}-L_{CG}),\qquad
K_\phi^{04}=K_{D\phi}^{04}+K_{\phi,\mathrm{est}}^{04}.
\]

`N_phi` é **calculado exclusivamente a partir do `Y_phi` recém-calculado** no mesmo bloco de código. Não existe uma segunda regressão independente de `N_phi`.

### Projeção completa 1996 — hipótese FSD da 28C.5

\[
Y_\phi^{96}=2F'_\phi\sin\beta+2F_0\cos\beta,\quad
N_\phi^{96}=Y_\phi^{96}(L_{CP}-L_{CG}).
\]

Esses coeficientes são originalmente projetados em FPU e transformados em FSD com `diag(-1,+1,-1)` aplicado ao vetor `[Y,K,N]`. A montagem reproduz literalmente a **28C.5** e não altera seu código.

**Não se deve comparar o sinal de Y_phi na 28C.7 (projeção FPU) com o de `heel_column[0]` (projeção FSD) sem aplicar a transformação.**

## R5 — critérios de integração no caso real

Execute usando o **mesmo** `boat` previamente resolvido que originou os números de 1600 kg/12 m/s:

```python
from openplaning.stability import assert_28c7_reference
result = assert_28c7_reference(boat)
print(result['audit'])
print(result['full_projection_eigenvalues'])
print(result['tabulated_eigenvalues'])
```

Valores **literais** tomados da auditoria anterior do usuário (as três asserções são efetivamente feitas na função):

| Origem | Grandeza | Valor esperado |
|---|---|---:|
| 28C.7, Tabela 6.3 | Y_phi | −8.867,644 N/rad |
| 28C.7, Tabela 6.3 | N_phi | −181,9909 N·m/rad |
| 28C.7, projeção completa FPU | Y_phi | +6.950,227 N/rad |
| 28C.10, tabulada FSD | Y_phi | −8.867,644 N/rad |
| 28C.10, tabulada FSD | N_phi | −181,9909 N·m/rad |
| 28C.10, projeção completa FSD | Y_phi | −6.950,227 N/rad |

A função verifica massa 1600 kg e velocidade 12 m/s, e usa `rtol=2e-5`, `atol=0.1` por padrão, devido às informações de entrada arredondadas neste log. Estes testes **não** foram executados localmente sem o estado operacional completo do `boat`. Não inventamos valores de `L_C`, `LCP`, `VCG`, `Ixx`, `Izz` do barco real.

A regressão do **autovalor** da variante de projeção completa contra a 28C.5 é realizada, bit a bit, no teste `test_full_projection_exactly_equals_28c5`. O teste `test_tabulated_eigenvalue_literal_regression` possui quatro raízes complexas **literais** de um estado sintético independente do caso R5, assim como `Y_phi`, `K_phi` e `N_phi` literais. O script R5 retorna os autovalores das duas versões para registro do usuário. Não há autovalor 28C.7 tabulado para o caso real: aquela etapa era uma auditoria de coeficientes, não uma matriz tabulada; seria incorreto fabricar tal referência.

## Instalação e testes

Extraia o ZIP na raiz da cópia de trabalho do repositório e confira o diff do `__init__.py`; somente acrescentamos importações, exportações e a reatribuição intencional de `from_resolved_planing_boat`.

```powershell
python -m unittest discover -s tests -p "test_source_variants.py" -v
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall -q openplaning
```

Com sucesso:

```powershell
git add openplaning/stability/source_variants.py openplaning/stability/__init__.py tests/test_source_variants.py docs/ETAPA_28C10_VARIANTES_EXPLICITAS.md
git commit -m "feat: require explicit source-labeled Lewandowski formulations"
git push origin feature/mass-properties
```

## Estado científico

1. **Fidelidade documental:** teste das equações impressas, consistente com o 28C.7.
2. **Consistência matemática:** regressão da montagem 28C.5, dos coeficientes e dos autovalores em fixture independente, além dos testes anteriores.
3. **Validade física/preditiva:** **não estabelecida**. Os números obtidos são diagnósticos de implementação; não usar para alegar margem de segurança.

Pendências: justificativa física da separação de forças e momentos entre 1996/2004, comparação com dados independentes, extensão aos apêndices, hipóteses geométricas e modelo 6 GDL quando necessário.
