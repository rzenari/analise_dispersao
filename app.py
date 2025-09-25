# ==============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import folium
from streamlit_folium import st_folium
from shapely.geometry import Polygon
import io
import os
import glob
import zipfile
import requests
from datetime import datetime
import json
from branca.colormap import linear
from collections import defaultdict
import ast


# ==============================================================================
# 2. CONFIGURAÇÃO DA PÁGINA E TÍTulos
# ==============================================================================
st.set_page_config(layout="wide", page_title="Análise de Dispersão Geográfica")

st.title("🗺️ Ferramenta de Análise de Dispersão Geográfica")
st.write("Faça o upload da sua planilha de cortes para analisar a distribuição geográfica e identificar clusters")

# ==============================================================================
# 3. FUNÇÕES DE ANÁLISE
# ==============================================================================

@st.cache_data
def carregar_malha_improdutividade(caminho_arquivo='improdutividade_historica.geojson'):
    """Carrega a malha de improdutividade a partir de um arquivo GeoJSON."""
    if not os.path.exists(caminho_arquivo):
        st.warning(f"Aviso: Arquivo '{caminho_arquivo}' não encontrado. A camada de improdutividade não será exibida.")
        return None
    try:
        gdf = gpd.read_file(caminho_arquivo)
        # Garante que as colunas de data sejam datetime
        gdf['menor_data'] = pd.to_datetime(gdf['menor_data'], errors='coerce')
        gdf['maior_data'] = pd.to_datetime(gdf['maior_data'], errors='coerce')
        # Converte a string de top_motivos para um objeto python
        gdf['top_motivos'] = gdf['top_motivos'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
        st.sidebar.info("Malha histórica de improdutividade carregada.")
        return gdf
    except Exception as e:
        st.error(f"Erro ao carregar ou processar o arquivo '{caminho_arquivo}': {e}")
        return None

@st.cache_data
def carregar_kmls(pasta_projeto):
    """
    Varre a pasta, lê arquivos .kml e .kmz, corrige geometrias inválidas
    e retorna um dicionário de geometrias individuais e um log de depuração.
    """
    kml_files = glob.glob(os.path.join(pasta_projeto, '*.kml'))
    kmz_files = glob.glob(os.path.join(pasta_projeto, '*.kmz'))
    all_gis_files = kml_files + kmz_files

    if not all_gis_files:
        return None, pd.DataFrame([{'Arquivo': 'Nenhum arquivo .kml ou .kmz encontrado', 'Status': 'N/A'}])
    
    debug_log = []
    geometrias_individuais = {}
    
    for gis_file in all_gis_files:
        nome_arquivo = os.path.basename(gis_file)
        try:
            gdf_file = None
            if gis_file.lower().endswith('.kml'):
                gdf_file = gpd.read_file(gis_file, driver='KML')
            elif gis_file.lower().endswith('.kmz'):
                with zipfile.ZipFile(gis_file, 'r') as z:
                    kml_filename = next((name for name in z.namelist() if name.lower().endswith('.kml')), None)
                    if kml_filename:
                        with z.open(kml_filename) as kml_content:
                            gdf_file = gpd.read_file(kml_content, driver='KML')
            
            if gdf_file is None or gdf_file.empty:
                debug_log.append({'Arquivo': nome_arquivo, 'Status': '⚠️ Aviso', 'Erro': 'Nenhum polígono encontrado.'})
                continue

            gdf_file = gdf_file[gdf_file.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            
            geometrias_corrigidas = []
            for poligono in gdf_file.geometry:
                if not poligono.is_valid:
                    poligono_corrigido = poligono.buffer(0)
                    if poligono_corrigido.is_valid and not poligono_corrigido.is_empty:
                        geometrias_corrigidas.append(poligono_corrigido)
                else:
                    geometrias_corrigidas.append(poligono)
            
            if geometrias_corrigidas:
                geometrias_individuais[nome_arquivo] = gpd.GeoSeries(geometrias_corrigidas).union_all()
                debug_log.append({'Arquivo': nome_arquivo, 'Status': '✅ Sucesso'})
            else:
                debug_log.append({'Arquivo': nome_arquivo, 'Status': '❌ Falha', 'Erro': 'Geometria inválida, não foi possível corrigir.'})

        except Exception as e:
            debug_log.append({'Arquivo': nome_arquivo, 'Status': '❌ Falha', 'Erro': str(e)})

    if not geometrias_individuais:
        return None, pd.DataFrame(debug_log)

    return geometrias_individuais, pd.DataFrame(debug_log)


def carregar_dados_completos(arquivo_enviado):
    """Lê o arquivo completo com todas as colunas."""
    arquivo_enviado.seek(0)
    
    def processar_dataframe(df):
        df.columns = df.columns.str.lower().str.strip()
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            st.error("ERRO: Colunas 'latitude' e/ou 'longitude' não foram encontradas."); st.write("Colunas encontradas:", df.columns.tolist())
            return None
        df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
        df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)
        df = df.dropna(subset=['latitude', 'longitude'])
        return df

    try:
        df = pd.read_csv(arquivo_enviado, encoding='utf-16', sep='\t')
        st.success("Arquivo CSV de cortes lido com sucesso (codificação: utf-16)."); return processar_dataframe(df)
    except Exception:
        try:
            arquivo_enviado.seek(0)
            df = pd.read_csv(arquivo_enviado, encoding='latin-1', sep=None, engine='python')
            st.success("Arquivo CSV de cortes lido com sucesso (codificação: latin-1)."); return processar_dataframe(df)
        except Exception:
            try:
                arquivo_enviado.seek(0)
                df = pd.read_excel(arquivo_enviado, engine='openpyxl')
                st.success("Arquivo Excel de cortes lido com sucesso."); return processar_dataframe(df)
            except Exception as e:
                st.error(f"Não foi possível ler o arquivo de cortes. Último erro: {e}"); return None

def carregar_dados_metas(arquivo_metas):
    """Lê o arquivo opcional de metas e equipes."""
    if arquivo_metas is None:
        return None
    arquivo_metas.seek(0)
    try:
        df = pd.read_excel(arquivo_metas, engine='openpyxl')
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        required_cols = ['centro_operativo', 'equipes', 'serviços_designados', 'produção', 'meta_diária']
        if all(col in df.columns for col in required_cols):
            df['centro_operativo'] = df['centro_operativo'].str.strip().str.upper()
            return df
        else:
            st.error(f"ERRO: A planilha de metas precisa conter as colunas: {', '.join(required_cols)}.")
            return None
    except Exception as e:
        st.error(f"Não foi possível ler a planilha de metas. Erro: {e}")
        return None

def executar_dbscan(gdf, eps_km=0.5, min_samples=3):
    """Executa o DBSCAN para encontrar clusters."""
    if gdf.empty or len(gdf) < min_samples: 
        gdf['cluster'] = -1
        return gdf
    
    gdf_copy = gdf.copy()
    raio_terra_km = 6371; eps_rad = eps_km / raio_terra_km
    coords = np.radians(gdf_copy[['latitude', 'longitude']].values)
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(coords)
    gdf_copy['cluster'] = db.labels_
    return gdf_copy

def simular_pacotes_por_densidade(gdf_co, n_equipes, capacidade_designada):
    """
    Simula pacotes com base na densidade, respeitando estritamente a capacidade de serviços designados.
    """
    if gdf_co.empty or n_equipes == 0 or capacidade_designada == 0:
        return gpd.GeoDataFrame(), gdf_co.copy()

    gdf_clusters_reais = gdf_co.copy()
    pacotes_candidatos = []
    
    for cluster_id in gdf_clusters_reais['cluster'].unique():
        if cluster_id == -1: continue
        gdf_cluster_atual = gdf_clusters_reais[gdf_clusters_reais['cluster'] == cluster_id]
        contagem = len(gdf_cluster_atual)

        if 0 < contagem <= capacidade_designada:
            pacotes_candidatos.append({'indices': gdf_cluster_atual.index, 'pontos': gdf_cluster_atual})
        
        elif contagem > capacidade_designada:
            gdf_temp = gdf_cluster_atual.copy()
            while len(gdf_temp) > 0:
                if len(gdf_temp) <= capacidade_designada:
                    pacotes_candidatos.append({'indices': gdf_temp.index, 'pontos': gdf_temp})
                    break
                
                coords_temp = gdf_temp[['longitude', 'latitude']].values
                tree = cKDTree(coords_temp)
                
                _, indices_vizinhos = tree.query(coords_temp[0], k=min(capacidade_designada, len(coords_temp)))
                
                indices_reais_no_gdf_temp = gdf_temp.index[indices_vizinhos]
                sub_pacote = gdf_temp.loc[indices_reais_no_gdf_temp]
                
                pacotes_candidatos.append({'indices': sub_pacote.index, 'pontos': sub_pacote})
                
                gdf_temp.drop(indices_reais_no_gdf_temp, inplace=True)

    pacotes_ranqueados = []
    for candidato in pacotes_candidatos:
        pontos = candidato['pontos']
        contagem = len(pontos)
        if contagem > 0:
            try:
                hull = pontos.unary_union.convex_hull
                area_km2 = (gpd.GeoDataFrame(geometry=[hull], crs="EPSG:4326").to_crs("EPSG:3857").geometry.area / 1_000_000).iloc[0]
                densidade = contagem / area_km2 if area_km2 > 0 else 0
                
                candidato['contagem'] = contagem
                candidato['area_km2'] = round(area_km2, 2)
                candidato['densidade'] = round(densidade, 2)
                pacotes_ranqueados.append(candidato)
            except Exception:
                continue

    pacotes_ranqueados.sort(key=lambda p: p['densidade'], reverse=True)
    pacotes_vencedores = pacotes_ranqueados[:n_equipes]
    
    indices_alocados = []
    gdf_co['pacote_id'] = -1 
    for i, pacote in enumerate(pacotes_vencedores):
        gdf_co.loc[pacote['indices'], 'pacote_id'] = i
        indices_alocados.extend(pacote['indices'])

    gdf_alocados = gdf_co.loc[indices_alocados].copy()
    gdf_excedentes = gdf_co.drop(indices_alocados).copy()

    return gdf_alocados, gdf_excedentes

def calcular_qualidade_carteira(row):
    """Calcula a qualidade da carteira com base nas metas e serviços agrupados."""
    meta_diaria = row.get('meta_diária', 0)
    if pd.isna(meta_diaria) or meta_diaria == 0: return "Sem Meta"
    n_agrupados = row.get('Agrupado', 0)
    total_servicos = row.get('total', 0)
    
    if n_agrupados >= meta_diaria: return "✅ Ótima"
    elif total_servicos >= meta_diaria: return "⚠️ Atenção"
    else: return "❌ Crítica"

def df_to_excel(df):
    """Converte um DataFrame para um objeto BytesIO em formato Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Dados')
    return output.getvalue()

@st.cache_data(ttl=600) # Cache de 10 minutos para o tempo atual
def get_current_weather(lat, lon, api_key):
    """Busca o tempo atual da API."""
    URL = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=pt_br"
    try:
        response = requests.get(URL)
        response.raise_for_status()
        data = response.json()
        return {
            "time": datetime.fromtimestamp(data['dt']).strftime('%H:%M'),
            "condition": data['weather'][0]['description'].title(),
            "icon": f"https://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png",
            "wind_speed_kmh": round(data['wind']['speed'] * 3.6, 1),
        }
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_weather_forecast(lat, lon, api_key):
    """Busca a previsão de 5 dias da API e estrutura por período."""
    URL = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=pt_br"
    try:
        response = requests.get(URL)
        response.raise_for_status()
        data = response.json()
        
        today_str = datetime.now().strftime('%Y-%m-%d')
        daily_data = {}
        
        for forecast in data['list']:
            dt_obj = datetime.fromtimestamp(forecast['dt'])
            date_key = dt_obj.strftime('%Y-%m-%d')
            hour = dt_obj.hour

            if date_key not in daily_data:
                daily_data[date_key] = { 'rain_madrugada': 0, 'manha_forecast': None, 'tarde_forecast': None }

            if date_key == today_str and hour in [0, 3, 6]:
                daily_data[date_key]['rain_madrugada'] += forecast.get('rain', {}).get('3h', 0)
            
            if hour == 9: daily_data[date_key]['manha_forecast'] = forecast
            if hour == 15: daily_data[date_key]['tarde_forecast'] = forecast

        forecast_list = []
        if today_str in daily_data:
            day_data = {'date': datetime.strptime(today_str, '%Y-%m-%d').strftime('%d/%m'), 'rain_madrugada': round(daily_data[today_str]['rain_madrugada'], 1), 'is_today': True}
            forecast_list.append(day_data)
        
        for date, periods in sorted(daily_data.items()):
            if date == today_str:
                continue

            day_data = {'date': datetime.strptime(date, '%Y-%m-%d').strftime('%d/%m'), 'is_today': False}
            for period_name, forecast_data in [('manha', periods['manha_forecast']), ('tarde', periods['tarde_forecast'])]:
                if forecast_data:
                    day_data[period_name] = {
                        "hour": datetime.fromtimestamp(forecast_data['dt']).strftime('%H:%M'),
                        "condition": forecast_data['weather'][0]['description'].title(),
                        "icon": f"https://openweathermap.org/img/wn/{forecast_data['weather'][0]['icon']}@2x.png",
                        "wind_speed_kmh": round(forecast_data['wind']['speed'] * 3.6, 1),
                        "rain_mm": round(forecast_data.get('rain', {}).get('3h', 0), 1)
                    }
                else: day_data[period_name] = None
            forecast_list.append(day_data)
        
        return forecast_list[:5]
    except Exception as e: return f"Erro ao buscar dados: {e}"

def get_operational_status(condition, wind_speed):
    """Define o status da operação com base no clima do período."""
    is_rainy = any(keyword in condition.lower() for keyword in ["chuva", "tempestade", "chuvisco"])
    is_windy = wind_speed > 40.0
    if is_rainy or is_windy: return "⚠️ Possível Contingência"
    else: return "✅ Operação Normal"

def desenhar_camadas_kml(map_object, risco_dict, laranja_dict):
    """Função auxiliar para desenhar as camadas KML em um mapa Folium."""
    if risco_dict:
        for nome, poligono in risco_dict.items():
            folium.GeoJson(poligono, style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': 2, 'fillOpacity': 0.2}, tooltip=f"Área de Risco: {nome}").add_to(map_object)
    if laranja_dict:
        for nome, poligono in laranja_dict.items():
            if poligono and not poligono.is_empty:
                folium.GeoJson(poligono, style_function=lambda x: {'fillColor': 'orange', 'color': 'orange', 'weight': 2, 'fillOpacity': 0.2}, tooltip=f"Área Laranja: {nome}").add_to(map_object)

def desenhar_camada_improdutividade(map_object, gdf_improd):
    """Desenha a camada de calor (heatmap) de improdutividade no mapa."""
    if gdf_improd is None or gdf_improd.empty:
        return
    
    # Cria uma cópia para evitar modificar o dataframe original
    gdf_for_map = gdf_improd.copy()

    # Converte colunas de Timestamp para string, pois não são serializáveis para JSON
    if 'menor_data' in gdf_for_map.columns:
        gdf_for_map['menor_data'] = gdf_for_map['menor_data'].dt.strftime('%d/%m/%Y')
    if 'maior_data' in gdf_for_map.columns:
        gdf_for_map['maior_data'] = gdf_for_map['maior_data'].dt.strftime('%d/%m/%Y')

    # Define a escala de cores de verde (0%) para vermelho (100%)
    colormap = linear.YlOrRd_09.scale(0, 100)
    colormap.caption = 'Taxa de Improdutividade Histórica (%)'
    
    # Adiciona a camada de hexágonos coloridos
    folium.GeoJson(
        gdf_for_map, # Usa a cópia com as datas formatadas como texto
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['taxa_improdutividade_%']),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.5
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['taxa_improdutividade_%', 'total_servicos', 'menor_data', 'maior_data'],
            aliases=['Improdutividade (%):', 'Total de Serviços:', 'De:', 'Até:'],
            localize=True
        )
    ).add_to(map_object)
    
    # Adiciona a legenda de cores ao mapa
    map_object.add_child(colormap)

def gerar_tooltip_html(row):
    """Gera um HTML formatado para o tooltip com dados de improdutividade agregados."""
    if pd.isna(row.get('taxa_improdutividade_media')):
        return """
            <div style="font-family: sans-serif;">
                <strong>Dados de Improdutividade Histórica:</strong><br>
                Não disponíveis para esta área.
            </div>
        """

    improd_perc = row['taxa_improdutividade_media']
    total_serv = int(row['total_servicos_hist'])
    data_min = row['menor_data_hist'].strftime('%d/%m/%Y')
    data_max = row['maior_data_hist'].strftime('%d/%m/%Y')
    top_motivos = row['top_5_motivos_agregados']
    
    motivos_html = ""
    if top_motivos:
        for motivo, perc in top_motivos:
            motivos_html += f"<li>{motivo.strip()}: {perc:.1f}%</li>"
    else:
        motivos_html = "<li>N/A</li>"
        
    html = f"""
    <div style="font-family: sans-serif; max-width: 350px;">
        <h4 style="margin-bottom: 5px; margin-top: 5px; border-bottom: 1px solid grey;">Análise Histórica da Área</h4>
        <strong>Taxa de Improdutividade:</strong> {improd_perc:.1f}%<br>
        <strong>Total de Serviços no Período:</strong> {total_serv}<br>
        <strong>Período de Análise:</strong> {data_min} a {data_max}<br>
        <strong style="margin-top: 5px; display: block;">Top 5 Motivos de Improdutividade:</strong>
        <ul style="margin-top: 2px; padding-left: 20px;">
            {motivos_html}
        </ul>
    </div>
    """
    return html

# ==============================================================================
# 4. LÓGICA PRINCIPAL DA APLICAÇÃO
# ==============================================================================
st.sidebar.header("Controles")
uploaded_file = st.sidebar.file_uploader("1. Escolha a planilha de cortes", type=["csv", "xlsx", "xls"])
metas_file = st.sidebar.file_uploader("2. Escolha a planilha de metas (Opcional)", type=["xlsx", "xls"])

# Carrega a malha de improdutividade uma vez
gdf_improd = carregar_malha_improdutividade()

if uploaded_file is not None:
    df_completo_original = carregar_dados_completos(uploaded_file)
    df_metas = carregar_dados_metas(metas_file)
    
    if df_completo_original is not None:
        st.sidebar.success(f"{len(df_completo_original)} registros carregados!")
        
        geometrias_kml_dict, kml_debug_log = carregar_kmls('.')
        if kml_debug_log is not None:
            sucesso_count = (kml_debug_log['Status'] == '✅ Sucesso').sum()
            st.sidebar.info(f"{sucesso_count} arquivo(s) KML/KMZ carregado(s) com sucesso.")
            with st.sidebar.expander("🔍 Depurador de Arquivos KML/KMZ"):
                st.dataframe(kml_debug_log)
        
        if df_metas is not None: 
            st.sidebar.info(f"Metas carregadas para {len(df_metas)} COs.")

        areas_sem_laranja = []
        if geometrias_kml_dict:
            st.sidebar.markdown("### Controle da Área Laranja")
            nomes_areas = list(geometrias_kml_dict.keys())
            areas_sem_laranja = st.sidebar.multiselect('Desativar Área Laranja para:', nomes_areas, help="Selecione as áreas de risco que NÃO devem ter a área laranja de 120m ao redor.")

        kml_laranja_dict = {}
        if geometrias_kml_dict:
            for nome_arquivo, poligono in geometrias_kml_dict.items():
                if nome_arquivo not in areas_sem_laranja:
                    geometria_proj = gpd.GeoSeries([poligono], crs="EPSG:4326").to_crs("EPSG:3857")
                    buffer_grande = geometria_proj.buffer(120)
                    geometria_laranja_proj = buffer_grande.difference(geometria_proj)
                    kml_laranja_dict[nome_arquivo] = gpd.GeoSeries(geometria_laranja_proj, crs="EPSG:3857").to_crs("EPSG:4326").union_all()

        st.sidebar.markdown("### Filtros da Análise")
        filtros = ['sucursal', 'centro_operativo', 'corte_recorte', 'prioridade']
        valores_selecionados = {}
        for coluna in filtros:
            if coluna in df_completo_original.columns:
                lista_unica = df_completo_original[coluna].dropna().unique().tolist()
                opcoes = sorted([str(item) for item in lista_unica])
                if coluna == 'prioridade':
                    valores_selecionados[coluna] = st.sidebar.multiselect(f"{coluna.replace('_', ' ').title()}", opcoes)
                else:
                    valores_selecionados[coluna] = st.sidebar.selectbox(f"{coluna.replace('_', ' ').title()}", ["Todos"] + opcoes)

        df_filtrado = df_completo_original.copy()
        for coluna, valor in valores_selecionados.items():
            if coluna in df_filtrado.columns:
                if coluna == 'prioridade':
                    if valor: df_filtrado = df_filtrado[df_filtrado[coluna].astype(str).isin(valor)]
                elif valor != "Todos": df_filtrado = df_filtrado[df_filtrado[coluna].astype(str) == valor]

        gdf_filtrado_base = gpd.GeoDataFrame(df_filtrado, geometry=gpd.points_from_xy(df_filtrado.longitude, df_filtrado.latitude), crs="EPSG:4326")
        
        gdf_filtrado_base['classificacao'] = 'A ser definido'
        gdf_filtrado_base['fonte_kml'] = ''

        if geometrias_kml_dict:
            for nome_arquivo, poligono in geometrias_kml_dict.items():
                indices_risco = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'A ser definido'].within(poligono)
                gdf_filtrado_base.loc[indices_risco[indices_risco].index, 'classificacao'] = 'Área de Risco'
                gdf_filtrado_base.loc[indices_risco[indices_risco].index, 'fonte_kml'] = nome_arquivo

        if kml_laranja_dict:
            for nome_arquivo, poligono_laranja in kml_laranja_dict.items():
                indices_laranja = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'A ser definido'].within(poligono_laranja)
                gdf_filtrado_base.loc[indices_laranja[indices_laranja].index, 'classificacao'] = 'Área Laranja'
                gdf_filtrado_base.loc[indices_laranja[indices_laranja].index, 'fonte_kml'] = nome_arquivo

        gdf_para_analise = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'A ser definido'].copy()
        
        st.sidebar.markdown("### Parâmetros de Cluster")
        eps_cluster_km = st.sidebar.slider("Raio do Cluster (km)", 0.1, 5.0, 1.0, 0.1)
        min_samples_cluster = st.sidebar.slider("Mínimo de Pontos por Cluster", 2, 20, 20, 1)

        if not gdf_para_analise.empty:
            gdf_com_clusters = executar_dbscan(gdf_para_analise, eps_km=eps_cluster_km, min_samples=min_samples_cluster)
            gdf_filtrado_base.loc[gdf_com_clusters[gdf_com_clusters['cluster'] != -1].index, 'classificacao'] = 'Agrupado'
            gdf_filtrado_base.loc[gdf_com_clusters[gdf_com_clusters['cluster'] == -1].index, 'classificacao'] = 'Disperso'
            gdf_filtrado_base = gdf_filtrado_base.merge(gdf_com_clusters[['cluster']], left_index=True, right_index=True, how='left')
        else: gdf_filtrado_base['cluster'] = -1
        
        gdf_filtrado_base['cluster'] = gdf_filtrado_base['cluster'].fillna(-1)
        gdf_filtrado_base.loc[gdf_filtrado_base['classificacao'] == 'A ser definido', 'classificacao'] = 'Disperso'
        
        gdf_risco = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Área de Risco'].copy()
        gdf_laranja = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Área Laranja'].copy()

        gdf_alocados_final = gpd.GeoDataFrame()
        if df_metas is not None:
            todos_alocados, todos_excedentes = [], []
            gdf_para_pacotes = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Agrupado'].copy()
            cos_filtrados = gdf_para_pacotes['centro_operativo'].unique() if not gdf_para_pacotes.empty else []
            for co in cos_filtrados:
                gdf_co = gdf_para_pacotes[gdf_para_pacotes['centro_operativo'] == co].copy()
                metas_co = df_metas[df_metas['centro_operativo'].str.strip().str.upper() == co.strip().upper()]
                if not metas_co.empty:
                    n_equipes = int(metas_co['equipes'].iloc[0])
                    capacidade_designada = int(metas_co['serviços_designados'].iloc[0])
                    if n_equipes > 0 and capacidade_designada > 0 and len(gdf_co) > 0:
                        alocados, excedentes_co = simular_pacotes_por_densidade(gdf_co, n_equipes, capacidade_designada)
                        todos_alocados.append(alocados)
                        todos_excedentes.append(excedentes_co)
                else: todos_excedentes.append(gdf_co)
            
            gdf_servicos_restantes = gdf_filtrado_base[gdf_filtrado_base['classificacao'] != 'Agrupado'].copy()
            if not gdf_servicos_restantes.empty: todos_excedentes.append(gdf_servicos_restantes)
            
            if todos_alocados:
                gdf_alocados_final = gpd.GeoDataFrame(pd.concat(todos_alocados, ignore_index=True), crs="EPSG:4326")
            if todos_excedentes:
                gdf_excedentes_final = gpd.GeoDataFrame(pd.concat(todos_excedentes, ignore_index=True), crs="EPSG:4326")

        st.header("Resultados da Análise")
        
        if not gdf_filtrado_base.empty:
            st.sidebar.markdown("### Filtro de Visualização do Mapa")
            opcoes_visualizacao = ["Todos", "Agrupado", "Disperso", "Área de Risco", "Área Laranja"]
            tipo_visualizacao = st.sidebar.radio("Mostrar nos mapas:", opcoes_visualizacao)
            
            st.sidebar.markdown("### 📥 Downloads")
            df_agrupados_download = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Agrupado'].drop(columns=['geometry', 'cluster', 'fonte_kml'], errors='ignore')
            if not df_agrupados_download.empty: st.sidebar.download_button(label="⬇️ Baixar Agrupados (Excel)", data=df_to_excel(df_agrupados_download), file_name='servicos_agrupados.xlsx')
            df_dispersos_download = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Disperso'].drop(columns=['geometry', 'cluster', 'fonte_kml'], errors='ignore')
            if not df_dispersos_download.empty: st.sidebar.download_button(label="⬇️ Baixar Dispersos (Excel)", data=df_to_excel(df_dispersos_download), file_name='servicos_dispersos.xlsx')
            if not gdf_risco.empty: st.sidebar.download_button(label="⬇️ Baixar Área de Risco (Excel)", data=df_to_excel(gdf_risco.drop(columns=['geometry', 'cluster'], errors='ignore')), file_name='servicos_area_risco.xlsx')
            if not gdf_laranja.empty: st.sidebar.download_button(label="⬇️ Baixar Área Laranja (Excel)", data=df_to_excel(gdf_laranja.drop(columns=['geometry', 'cluster'], errors='ignore')), file_name='servicos_area_laranja.xlsx')
            if not gdf_alocados_final.empty: st.sidebar.download_button(label="⬇️ Baixar Pacotes de Trabalho (Excel)", data=df_to_excel(gdf_alocados_final.drop(columns=['geometry', 'cluster'], errors='ignore')), file_name='pacotes_de_trabalho.xlsx')

            gdf_visualizacao = gdf_filtrado_base.copy()
            if tipo_visualizacao != "Todos": gdf_visualizacao = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == tipo_visualizacao]

            lista_abas = ["🗺️ Análise Geográfica", "📊 Resumo por CO", "📍 Contorno dos Clusters"]
            if df_metas is not None: lista_abas.append("📦 Pacotes de Trabalho")
            lista_abas.append("🌦️ Painel de Risco Climático")
            lista_abas.append("💡 Metodologia")
            tabs = st.tabs(lista_abas)
            
            tab_index = 0

            with tabs[tab_index]: # Análise Geográfica
                with st.spinner('Carregando análise e mapa...'):
                    st.subheader("Resumo da Análise de Classificação")
                    total_servicos, n_agrupados, n_dispersos, n_risco, n_laranja = len(gdf_filtrado_base), len(gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Agrupado']), len(gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Disperso']), len(gdf_risco), len(gdf_laranja)
                    p_agrupados, p_dispersos, p_risco, p_laranja = (n_agrupados / total_servicos * 100) if total_servicos > 0 else 0, (n_dispersos / total_servicos * 100) if total_servicos > 0 else 0, (n_risco / total_servicos * 100) if total_servicos > 0 else 0, (n_laranja / total_servicos * 100) if total_servicos > 0 else 0
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Nº Agrupados", f"{n_agrupados}", f"{p_agrupados:.1f}%"); col2.metric("Nº Dispersos", f"{n_dispersos}", f"{p_dispersos:.1f}%"); col3.metric("Nº em Área de Risco", f"{n_risco}", f"{p_risco:.1f}%"); col4.metric("Nº em Área Laranja", f"{n_laranja}", f"{p_laranja:.1f}%")
                    st.subheader(f"Mapa Interativo")
                    if not gdf_visualizacao.empty:
                        map_center = [gdf_visualizacao.latitude.mean(), gdf_visualizacao.longitude.mean()]
                        m = folium.Map(location=map_center, zoom_start=11)
                        desenhar_camadas_kml(m, geometrias_kml_dict, kml_laranja_dict)
                        cor_classificacao = {'Agrupado': 'blue', 'Disperso': 'gray', 'Área de Risco': 'red', 'Área Laranja': 'orange'}
                        for _, row in gdf_visualizacao.iterrows(): folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=5, color=cor_classificacao.get(row['classificacao'], 'black'), fill=True, fill_color=cor_classificacao.get(row['classificacao'], 'black'), fill_opacity=0.7, popup=f"Classificação: {row['classificacao']}").add_to(m)
                        st_folium(m, use_container_width=True, height=700)
                    else: st.warning("Nenhum serviço para exibir no mapa com os filtros atuais.")
            tab_index += 1

            with tabs[tab_index]: # Resumo por CO
                with st.spinner('Gerando tabela de resumo...'):
                    st.subheader("Resumo por Centro Operativo")
                    resumo_co = gdf_filtrado_base.groupby('centro_operativo')['classificacao'].value_counts().unstack(fill_value=0)
                    for col in ['Agrupado', 'Disperso', 'Área de Risco', 'Área Laranja']:
                        if col not in resumo_co.columns: resumo_co[col] = 0
                    resumo_co['total'] = resumo_co.sum(axis=1)
                    resumo_co['% Agrupado'] = (resumo_co['Agrupado'] / resumo_co['total'] * 100).round(1)
                    resumo_co['% Disperso'] = (resumo_co['Disperso'] / resumo_co['total'] * 100).round(1)
                    resumo_co['% Área de Risco'] = (resumo_co['Área de Risco'] / resumo_co['total'] * 100).round(1)
                    resumo_co['% Área Laranja'] = (resumo_co['Área Laranja'] / resumo_co['total'] * 100).round(1)
                    resumo_co.reset_index(inplace=True)
                    if df_metas is not None:
                        resumo_simulacao = gdf_alocados_final.groupby('centro_operativo').agg(Serviços_Alocados=('pacote_id', 'size'), Pacotes_Criados=('pacote_id', 'nunique')).reset_index()
                        resumo_co = pd.merge(resumo_co, resumo_simulacao, on='centro_operativo', how='left')
                        resumo_co['centro_operativo_join_key'] = resumo_co['centro_operativo'].str.strip().str.upper()
                        df_metas['centro_operativo_join_key'] = df_metas['centro_operativo'].str.strip().str.upper()
                        resumo_co = pd.merge(resumo_co, df_metas, on='centro_operativo_join_key', how='left')
                        if 'centro_operativo_x' in resumo_co.columns: resumo_co = resumo_co.drop(columns=['centro_operativo_y', 'centro_operativo_join_key']).rename(columns={'centro_operativo_x': 'centro_operativo'})
                        resumo_co['Expectativa_Execução'] = resumo_co['equipes'] * resumo_co['produção']
                        resumo_co['Aderência_à_Meta_%'] = (resumo_co['Serviços_Alocados'] / resumo_co['meta_diária'] * 100).fillna(0).round(1)
                        resumo_co['Ocupação_das_Equipes_%'] = (resumo_co['Pacotes_Criados'] / resumo_co['equipes'] * 100).fillna(0).round(1)
                        resumo_co['qualidade_da_carteira'] = resumo_co.apply(calcular_qualidade_carteira, axis=1)
                        cols_ordem = ['centro_operativo', 'total', 'Agrupado', '% Agrupado', 'Disperso', '% Disperso', 'Área de Risco', '% Área de Risco', 'Área Laranja', '% Área Laranja', 'equipes', 'meta_diária', 'Expectativa_Execução', 'Serviços_Alocados', 'Pacotes_Criados', 'Aderência_à_Meta_%', 'Ocupação_das_Equipes_%', 'qualidade_da_carteira']
                        cols_existentes = [col for col in cols_ordem if col in resumo_co.columns]
                        resumo_co = resumo_co[cols_existentes].fillna(0)
                    st.dataframe(resumo_co, use_container_width=True)
            tab_index += 1

            with tabs[tab_index]: # Contorno dos Clusters
                with st.spinner('Desenhando contornos e analisando improdutividade...'):
                    st.subheader("Contorno Geográfico dos Clusters com Análise Histórica")
                    st.write("Este mapa desenha um polígono ao redor de cada hotspot, sobreposto à malha histórica de improdutividade.")
                    gdf_clusters_reais = gdf_filtrado_base[(gdf_filtrado_base['classificacao'] == 'Agrupado') & (gdf_filtrado_base['cluster'] != -1)]
                    if not gdf_clusters_reais.empty:
                        map_center_hull = [gdf_clusters_reais.latitude.mean(), gdf_clusters_reais.longitude.mean()]
                        m_hull = folium.Map(location=map_center_hull, zoom_start=11)
                        
                        desenhar_camada_improdutividade(m_hull, gdf_improd)
                        desenhar_camadas_kml(m_hull, geometrias_kml_dict, kml_laranja_dict)

                        try:
                            hulls = gdf_clusters_reais.dissolve(by='cluster').convex_hull
                            gdf_hulls = gpd.GeoDataFrame(geometry=hulls).reset_index()
                            
                            counts = gdf_clusters_reais.groupby('cluster').size().rename('contagem')
                            gdf_hulls = gdf_hulls.merge(counts, on='cluster')
                            gdf_hulls_proj = gdf_hulls.to_crs("EPSG:3857")
                            gdf_hulls['area_km2'] = (gdf_hulls_proj.geometry.area / 1_000_000).round(2)
                            gdf_hulls['densidade'] = (gdf_hulls['contagem'] / gdf_hulls['area_km2']).replace([np.inf, -np.inf], 0).round(2)
                            
                            # Análise de Improdutividade
                            if gdf_improd is not None:
                                # >>> INÍCIO DA CORREÇÃO <<<
                                gdf_improd_renamed = gdf_improd.rename(columns={'geometry': 'geom_improd'}).set_geometry('geom_improd')
                                gdf_hulls_com_improd = gpd.sjoin(gdf_hulls, gdf_improd_renamed, how="left", predicate="intersects")
                                # >>> FIM DA CORREÇÃO <<<

                                # Agregação dos dados históricos para cada cluster
                                def aggregate_improd(df_group):
                                    if df_group['total_servicos'].isnull().all():
                                        return pd.Series({'taxa_improdutividade_media': np.nan, 'total_servicos_hist': 0, 'menor_data_hist': pd.NaT, 'maior_data_hist': pd.NaT, 'top_5_motivos_agregados': []})

                                    total_servicos_sum = df_group['total_servicos'].sum()
                                    if total_servicos_sum > 0:
                                        improd_ponderada = (df_group['taxa_improdutividade_%'] * df_group['total_servicos']).sum() / total_servicos_sum
                                    else:
                                        improd_ponderada = 0
                                        
                                    # Agregar motivos
                                    motivos_agg = defaultdict(float)
                                    for lista_motivos in df_group['top_motivos'].dropna():
                                        for motivo, perc in lista_motivos:
                                            motivos_agg[motivo] += perc
                                    
                                    top_5 = sorted(motivos_agg.items(), key=lambda item: item[1], reverse=True)[:5]
                                    
                                    return pd.Series({
                                        'taxa_improdutividade_media': improd_ponderada,
                                        'total_servicos_hist': total_servicos_sum,
                                        'menor_data_hist': df_group['menor_data'].min(),
                                        'maior_data_hist': df_group['maior_data'].max(),
                                        'top_5_motivos_agregados': top_5
                                    })
                                
                                resumo_improd = gdf_hulls_com_improd.groupby('cluster').apply(aggregate_improd)
                                gdf_hulls = gdf_hulls.merge(resumo_improd, on='cluster', how='left')
                                gdf_hulls['tooltip_html'] = gdf_hulls.apply(gerar_tooltip_html, axis=1)

                                tooltip_obj = folium.GeoJsonTooltip(fields=['tooltip_html'], aliases=[''], localize=True, sticky=True, style="""
                                    background-color: #F0EFEF;
                                    border: 2px solid black;
                                    border-radius: 3px;
                                    box-shadow: 3px;
                                """)
                            else:
                                tooltip_obj = folium.GeoJsonTooltip(fields=['contagem', 'area_km2', 'densidade'], aliases=['Nº de Cortes:', 'Área (km²):', 'Cortes/km²:'], localize=True, sticky=True)

                            folium.GeoJson(
                                gdf_hulls, 
                                style_function=lambda x: {'color': 'blue', 'weight': 2.5, 'fillColor': 'blue', 'fillOpacity': 0.15}, 
                                tooltip=tooltip_obj
                            ).add_to(m_hull)
                            st_folium(m_hull, use_container_width=True, height=700)
                        except Exception as e: st.error(f"Não foi possível desenhar os contornos ou analisar a improdutividade. Erro: {e}")
                    else: st.warning("Nenhum cluster para desenhar.")
            tab_index += 1

            if df_metas is not None:
                with tabs[tab_index]: # Pacotes de Trabalho
                    st.subheader("Simulação de Pacotes de Trabalho com Análise Histórica")
                    st.write("Visualização dos pacotes de trabalho simulados sobre a malha histórica de improdutividade.")
                    if not gdf_alocados_final.empty:
                        with st.spinner('Gerando simulação e analisando improdutividade...'):
                            cos_simulados = gdf_alocados_final['centro_operativo'].unique()
                            metas_filtradas = df_metas[df_metas['centro_operativo'].isin(cos_simulados)]

                            if not metas_filtradas.empty:
                                # Painel de Métricas
                                equipes_disponiveis, meta_diaria_total = metas_filtradas['equipes'].sum(), metas_filtradas['meta_diária'].sum()
                                metas_filtradas['expectativa_execucao'] = metas_filtradas['equipes'] * metas_filtradas['produção']
                                expectativa_total = metas_filtradas['expectativa_execucao'].sum()
                                servicos_agrupados_para_pacotes, servicos_alocados = len(gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Agrupado']), len(gdf_alocados_final)
                                pacotes_criados = gdf_alocados_final['pacote_id'].nunique() if not gdf_alocados_final.empty else 0
                                servicos_excedentes = len(gdf_excedentes_final) if 'gdf_excedentes_final' in locals() else len(gdf_filtrado_base) - servicos_alocados
                                aderencia_meta = (servicos_alocados / meta_diaria_total * 100) if meta_diaria_total > 0 else 0
                                ocupacao_equipes = (pacotes_criados / equipes_disponiveis * 100) if equipes_disponiveis > 0 else 0
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown("##### Planejamento"); st.metric("Equipes", f"{int(equipes_disponiveis)}"); st.metric("Meta Diária", f"{int(meta_diaria_total)}"); st.metric("Expectativa Execução", f"{int(expectativa_total)}")
                                with col2:
                                    st.markdown("##### Simulação"); st.metric("Serv. Agrupados", f"{servicos_agrupados_para_pacotes}"); st.metric("Serv. Alocados", f"{servicos_alocados}"); st.metric("Pacotes Criados", f"{pacotes_criados}"); st.metric("Serv. Excedentes", f"{servicos_excedentes}")
                                with col3:
                                    st.markdown("##### Desempenho"); st.metric("Aderência à Meta", f"{aderencia_meta:.1f}%"); st.metric("Ocupação das Equipes", f"{ocupacao_equipes:.1f}%")
                                st.markdown("---")
                            
                            # Mapa de Pacotes
                            map_center_pacotes = [gdf_filtrado_base.latitude.mean(), gdf_filtrado_base.longitude.mean()]
                            m_pacotes = folium.Map(location=map_center_pacotes, zoom_start=10)
                            
                            desenhar_camada_improdutividade(m_pacotes, gdf_improd)
                            desenhar_camadas_kml(m_pacotes, geometrias_kml_dict, kml_laranja_dict)
                            
                            cores_co = {co: color for co, color in zip(gdf_filtrado_base['centro_operativo'].unique(), ['blue', 'green', 'purple', 'orange', 'darkred', 'red', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'lightgreen', 'pink', 'lightblue', 'lightgray', 'black'])}
                            
                            gdf_hulls_pacotes = gdf_alocados_final.dissolve(by=['centro_operativo', 'pacote_id']).convex_hull.reset_index().rename(columns={0: 'geometry'}).set_geometry('geometry')
                            counts_pacotes = gdf_alocados_final.groupby(['centro_operativo', 'pacote_id']).size().rename('contagem').reset_index()
                            gdf_hulls_pacotes = gdf_hulls_pacotes.merge(counts_pacotes, on=['centro_operativo', 'pacote_id'])
                            gdf_hulls_pacotes['area_km2'] = (gdf_hulls_pacotes.to_crs("EPSG:3857").geometry.area / 1_000_000).round(2)
                            
                            if gdf_improd is not None:
                                # >>> INÍCIO DA CORREÇÃO <<<
                                gdf_improd_renamed_pacotes = gdf_improd.rename(columns={'geometry': 'geom_improd'}).set_geometry('geom_improd')
                                gdf_pacotes_com_improd = gpd.sjoin(gdf_hulls_pacotes, gdf_improd_renamed_pacotes, how="left", predicate="intersects")
                                # >>> FIM DA CORREÇÃO <<<

                                resumo_improd_pacotes = gdf_pacotes_com_improd.groupby(['centro_operativo', 'pacote_id']).apply(aggregate_improd)
                                gdf_hulls_pacotes = gdf_hulls_pacotes.merge(resumo_improd_pacotes, on=['centro_operativo', 'pacote_id'], how='left')
                                gdf_hulls_pacotes['tooltip_html'] = gdf_hulls_pacotes.apply(gerar_tooltip_html, axis=1)

                                tooltip_pacotes = folium.GeoJsonTooltip(fields=['tooltip_html'], aliases=[''], localize=True, sticky=True, style="""
                                    background-color: #F0EFEF;
                                    border: 2px solid black;
                                    border-radius: 3px;
                                    box-shadow: 3px;
                                """)
                            else:
                                tooltip_pacotes = folium.GeoJsonTooltip(fields=['centro_operativo', 'pacote_id', 'contagem', 'area_km2'], aliases=['CO:', 'Pacote:', 'Nº de Serviços:', 'Área (km²):'], localize=True, sticky=True)

                            folium.GeoJson(gdf_hulls_pacotes, style_function=lambda feature: {'color': cores_co.get(feature['properties']['centro_operativo'], 'gray'), 'weight': 2.5, 'fillColor': cores_co.get(feature['properties']['centro_operativo'], 'gray'), 'fillOpacity': 0.25}, tooltip=tooltip_pacotes).add_to(m_pacotes)
                            for _, row in gdf_alocados_final.iterrows(): folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=3, color=cores_co.get(row['centro_operativo'], 'gray'), fill=True, fill_opacity=1, popup=f"Pacote: {row['pacote_id']}").add_to(m_pacotes)
                            st_folium(m_pacotes, use_container_width=True, height=700)
                    else: st.info("Nenhum pacote de trabalho foi gerado para simular.")
                tab_index += 1
            
            with tabs[tab_index]: # Painel de Risco Climático
                st.subheader("Painel de Risco Climático")
                api_key_forecast = st.secrets.get("OPENWEATHER_API_KEY")
                if not api_key_forecast:
                    st.error("Chave da API OpenWeatherMap (Forecast) não encontrada.")
                else:
                    centroids = gdf_filtrado_base.dissolve(by='centro_operativo').centroid
                    co_coords = {co: (point.y, point.x) for co, point in centroids.items()}
                    for co, coords in co_coords.items():
                        with st.expander(f"**{co}**"):
                            forecast_data = get_weather_forecast(coords[0], coords[1], api_key_forecast)
                            if isinstance(forecast_data, list) and forecast_data:
                                for day in forecast_data:
                                    if day.get('is_today', False):
                                        st.markdown(f"**Hoje ({day['date']})**")
                                        current_weather = get_current_weather(coords[0], coords[1], api_key_forecast)
                                        if current_weather:
                                            st.markdown(f"**Condições Atuais ({current_weather['time']})**"); st.image(current_weather['icon'], width=60); st.markdown(current_weather['condition']); st.markdown(f"Vento: **{current_weather['wind_speed_kmh']} km/h**")
                                            is_rainy_now = any(k in current_weather['condition'].lower() for k in ["chuva", "tempestade"])
                                            is_windy_now = current_weather['wind_speed_kmh'] > 40.0
                                            heavy_overnight_rain = day.get('rain_madrugada', 0) > 5.0
                                            if is_rainy_now or is_windy_now or heavy_overnight_rain:
                                                st.markdown("⚠️ **Possível Contingência**")
                                                if heavy_overnight_rain and not(is_rainy_now or is_windy_now): st.caption("*(Impacto por chuva na madrugada)*")
                                            else: st.markdown("✅ **Operação Normal**")
                                        else: st.warning("Não foi possível obter o tempo atual.")
                                    else:
                                        st.markdown(f"**{day['date']}**")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            manha = day.get('manha')
                                            if manha:
                                                st.markdown(f"**Manhã ({manha['hour']})**"); st.image(manha['icon'], width=60); st.markdown(manha['condition']); st.markdown(f"Vento: **{manha['wind_speed_kmh']} km/h**")
                                                if manha['rain_mm'] > 0: st.markdown(f"💧 Chuva: **{manha['rain_mm']} mm**")
                                                st.markdown(get_operational_status(manha['condition'], manha['wind_speed_kmh']))
                                            else: st.info("Dados de Manhã indisponíveis.")
                                        with col2:
                                            tarde = day.get('tarde')
                                            if tarde:
                                                st.markdown(f"**Tarde ({tarde['hour']})**"); st.image(tarde['icon'], width=60); st.markdown(tarde['condition']); st.markdown(f"Vento: **{tarde['wind_speed_kmh']} km/h**")
                                                if tarde['rain_mm'] > 0: st.markdown(f"💧 Chuva: **{tarde['rain_mm']} mm**")
                                                st.markdown(get_operational_status(tarde['condition'], tarde['wind_speed_kmh']))
                                            else: st.info("Dados de Tarde indisponíveis.")
                                    st.markdown("<hr>", unsafe_allow_html=True)
                            else: st.warning(f"Não foi possível obter a previsão. Erro: {forecast_data}")
            tab_index += 1

            with tabs[tab_index]: # Metodologia
                st.subheader("As Metodologias por Trás da Análise")
                st.markdown("""
                - **Detecção de Áreas de Exceção (KML/KMZ):** ...
                - **Detecção de Hotspots (DBSCAN):** ...
                - **Simulação de Pacotes (Ranking de Densidade):** ...
                - **Painel de Risco Climático:** Para o dia de **hoje**, a ferramenta busca as **condições em tempo real** para fornecer a visão mais precisa possível. Para os **dias futuros**, a exibição é dividida em "Manhã" (09:00) e "Tarde" (15:00). A lógica de contingência considera:
                    - **Análise da Madrugada:** ...
                    - **Análise Presente e Futura:** ...
                
                Um alerta de "⚠️ **Possível Contingência**" para a manhã é acionado se for identificada chuva na madrugada ou condições de chuva/vento forte no período da manhã.
                """)
                st.subheader("Perguntas Frequentes (FAQ)")
                st.markdown("""
                - **Qual a diferença entre as colunas da planilha de metas?** ...
                - **Qual a estratégia usada para formar os pacotes de trabalho?** ...
                """)
        else: st.warning("Nenhum dado para exibir com os filtros atuais.")
else: st.info("Aguardando o upload de um arquivo para iniciar a análise.")
