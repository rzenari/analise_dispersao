# ==============================================================================
# 3. FUNÇÕES DE ANÁLISE (COM CACHE PARA PERFORMANCE)
# ==============================================================================

# O cache do Streamlit guarda o resultado da função. Se a função for chamada
# com os mesmos parâmetros, ele retorna o resultado guardado sem reexecutar.
@st.cache_data
def carregar_dados(arquivo_enviado):
    """Lê o arquivo CSV ou Excel, tentando diferentes codificações e separadores."""
    
    # Lista de codificações comuns para tentar
    encodings_to_try = ['utf-8-sig', 'latin-1', 'utf-8', 'iso-8859-1']
    
    # Tenta ler como CSV com diferentes codificações
    for encoding in encodings_to_try:
        try:
            # O seek(0) é importante para "rebobinar" o arquivo a cada tentativa
            arquivo_enviado.seek(0)
            df = pd.read_csv(arquivo_enviado, encoding=encoding, sep=None, engine='python')
            st.success(f"Arquivo CSV lido com sucesso usando a codificação: {encoding}")
            
            # Validação de colunas essenciais
            if 'latitude' not in df.columns or 'longitude' not in df.columns:
                st.error("O arquivo precisa conter as colunas 'latitude' e 'longitude'.")
                return None

            # Limpeza dos dados de coordenadas (substitui vírgula por ponto)
            df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
            df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)
            df = df.dropna(subset=['latitude', 'longitude'])
            return df
        except Exception:
            continue # Se falhar, tenta a próxima codificação

    # Se todas as tentativas de CSV falharem, tenta ler como Excel
    try:
        arquivo_enviado.seek(0)
        df = pd.read_excel(arquivo_enviado, engine='openpyxl') # Especificando o 'engine'
        st.success("Arquivo lido com sucesso como Excel (.xlsx).")

        # Validação e limpeza (repetido para o caso de ser Excel)
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            st.error("O arquivo precisa conter as colunas 'latitude' e 'longitude'.")
            return None
        df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
        df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)
        df = df.dropna(subset=['latitude', 'longitude'])
        return df
    except Exception as e:
        st.error(f"Não foi possível ler o arquivo como CSV ou Excel. Último erro: {e}")
        return None
