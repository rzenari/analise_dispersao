@st.cache_data
def carregar_dados(arquivo_enviado):
    """Lê o arquivo de forma otimizada, carregando apenas as colunas necessárias."""
    
    # Lista de colunas que realmente vamos usar. Isso economiza MUITA memória.
    colunas_necessarias = [
        'latitude', 'longitude', 'sucursal', 
        'centro_operativo', 'corte_recorte', 'prioridade'
    ]
    
    df = None
    encodings_to_try = ['utf-16', 'utf-8-sig', 'latin-1', 'utf-8']
    
    def processar_dataframe(df):
        """Função auxiliar para limpar e processar o dataframe."""
        df.columns = df.columns.str.lower().str.strip()
        
        # Verifica se todas as colunas necessárias foram carregadas
        if not all(col in df.columns for col in colunas_necessarias):
            st.error("ERRO: Uma ou mais colunas essenciais (latitude, longitude, sucursal, etc.) não foram encontradas no arquivo.")
            st.write("Colunas necessárias:", colunas_necessarias)
            st.write("Colunas encontradas:", df.columns.tolist())
            return None

        df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
        df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)
        df = df.dropna(subset=['latitude', 'longitude'])
        return df

    # Tenta ler como CSV
    for encoding in encodings_to_try:
        try:
            arquivo_enviado.seek(0)
            df = pd.read_csv(
                arquivo_enviado, 
                encoding=encoding, 
                sep=None, 
                engine='python',
                usecols=lambda column: column.strip().lower() in colunas_necessarias
            )
            st.success(f"Arquivo CSV lido com sucesso (codificação: {encoding}).")
            return processar_dataframe(df)
        except Exception:
            continue
    
    # Se CSV falhou, tenta ler como Excel
    try:
        arquivo_enviado.seek(0)
        # No Excel, não podemos usar 'usecols' da mesma forma antes de ler,
        # então lemos tudo e filtramos depois. Pode falhar se o Excel for muito grande.
        df_excel = pd.read_excel(arquivo_enviado, engine='openpyxl')
        st.success("Arquivo lido com sucesso como Excel (.xlsx).")
        
        # Padroniza as colunas antes de filtrar
        df_excel.columns = df_excel.columns.str.lower().str.strip()
        colunas_encontradas = [col for col in colunas_necessarias if col in df_excel.columns]
        df = df_excel[colunas_encontradas]
        
        return processar_dataframe(df)
    except Exception as e:
        st.error(f"Não foi possível ler o arquivo. Último erro: {e}")
        return None
