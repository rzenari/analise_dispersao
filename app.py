@st.cache_data
def carregar_dados(arquivo_enviado):
    """Lê o arquivo CSV ou Excel, tentando diferentes codificações e padronizando os cabeçalhos."""
    df = None
    encodings_to_try = ['utf-8-sig', 'latin-1', 'utf-8', 'iso-8859-1']
    
    # Tenta ler como CSV
    for encoding in encodings_to_try:
        try:
            arquivo_enviado.seek(0)
            df = pd.read_csv(arquivo_enviado, encoding=encoding, sep=None, engine='python')
            st.success(f"Arquivo CSV lido com sucesso usando a codificação: {encoding}")
            break 
        except Exception:
            continue
    
    # Se CSV falhou, tenta ler como Excel
    if df is None:
        try:
            arquivo_enviado.seek(0)
            df = pd.read_excel(arquivo_enviado, engine='openpyxl')
            st.success("Arquivo lido com sucesso como Excel (.xlsx).")
        except Exception as e:
            st.error(f"Não foi possível ler o arquivo como CSV ou Excel. Último erro: {e}")
            return None

    # ==============================================================================
    # LINHAS NOVAS E IMPORTANTES: PADRONIZAÇÃO DOS NOMES DAS COLUNAS
    # ==============================================================================
    df.columns = df.columns.str.lower().str.strip()
    # A linha acima converte tudo para minúsculas e remove espaços no início/fim.
    # Ex: ' Latitude ' se torna 'latitude'.
    # ==============================================================================

    # Agora a verificação vai funcionar, não importa o formato original
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.error("ERRO CRÍTICO: As colunas 'latitude' e/ou 'longitude' não foram encontradas mesmo após a padronização. Verifique o arquivo de origem.")
        st.write("Colunas encontradas:", df.columns.tolist()) # Ajuda a debugar
        return None

    try:
        df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
        df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)
        df = df.dropna(subset=['latitude', 'longitude'])
        return df
    except Exception as e:
        st.error(f"Erro ao converter as colunas de latitude/longitude para números. Verifique se elas contêm apenas valores numéricos. Detalhe do erro: {e}")
        return None
