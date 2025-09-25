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
from collections import Counter

# ==============================================================================
# 2. CONFIGURAÇÃO DA PÁGINA E TÍTULOS
# ==============================================================================
st.set_page_config(layout="wide", page_title="Análise de Dispersão Geográfica")

st.title("🗺️ Ferramenta de Análise de Dispersão Geográfica")
st.write("Faça o upload da sua planilha de cortes para analisar a distribuição geográfica e identificar clusters")

# ==============================================================================
# 3. FUNÇÕES DE ANÁLISE
# ==============================================================================

@st.cache_data
def carregar_malha_improdutividade(filepath='improdutividade_historica.geojson'):
    """Carrega o arquivo GeoJSON pré-processado com a malha de improdutividade."""
    try:
        gdf = gpd.read_file(filepath)
        # Convertendo a string JSON de volta para uma lista de dicionários
        gdf['top_motivos'] = gdf['top_motivos'].apply(json.loads)
        return gdf
    except Exception:
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
        df['latitude'] = pd.to_numeric(df['latitude'].astype(str).str.replace(',', '.'), errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'].astype(str).str.replace(',', '.'), errors='coerce')
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

@st.cache_data(ttl=600)
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

def formatar_tooltip_improdutividade(row, prefix=""):
    """Formata o texto do tooltip com dados de improdutividade."""
    if pd.isna(row.get('taxa_improdutividade_%_media')):
        return ""
    
    texto = "<br>---<br><b>Improdutividade Histórica:</b><br>"
    texto += f" • Taxa Média: {row['taxa_improdutividade_%_media']:.1f}%<br>"
    texto += f" • Total de Serviços: {int(row['total_servicos_sum'])}<br>"
    texto += f" • Período: {row['menor_data_min']} a {row['maior_data_max']}<br>"
    
    motivos_agregados = row.get('top_motivos_agregados')
    if motivos_agregados and isinstance(motivos_agregados, list) and len(motivos_agregados) > 0:
        texto += "<b>Principais Motivos:</b><br>"
        for i, motivo in enumerate(motivos_agregados):
            texto += f" {i+1}. {motivo['motivo']} ({motivo['rep_%']:.1f}%)<br>"
            
    return texto.replace('"',"'")
        
# ==============================================================================
# 4. LÓGICA PRINCIPAL DA APLICAÇÃO
# ==============================================================================
st.sidebar.header("Controles")
uploaded_file = st.sidebar.file_uploader("1. Escolha a planilha de cortes", type=["csv", "xlsx", "xls"])
metas_file = st.sidebar.file_uploader("2. Escolha a planilha de metas (Opcional)", type=["xlsx", "xls"])

gdf_improdutividade = carregar_malha_improdutividade()
if gdf_improdutividade is None:
    st.sidebar.warning("Arquivo 'improdutividade_historica.geojson' não encontrado. A camada de improdutividade não será exibida.")

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
            
            if todos_alocados: gdf_alocados_final = gpd.GeoDataFrame(pd.concat(todos_alocados, ignore_index=True), crs="EPSG:4326")
            if todos_excedentes: gdf_excedentes_final = gpd.GeoDataFrame(pd.concat(todos_excedentes, ignore_index=True), crs="EPSG:4326")

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
                with st.spinner('Desenhando contornos dos clusters...'):
                    st.subheader("Contorno Geográfico dos Clusters (Hotspots)")
                    st.write("Este mapa desenha um polígono ao redor de cada hotspot da categoria 'Agrupado'.")
                    gdf_clusters_reais = gdf_filtrado_base[(gdf_filtrado_base['classificacao'] == 'Agrupado') & (gdf_filtrado_base['cluster'] != -1)]
                    if not gdf_clusters_reais.empty:
                        map_center_hull = [gdf_clusters_reais.latitude.mean(), gdf_clusters_reais.longitude.mean()]
                        m_hull = folium.Map(location=map_center_hull, zoom_start=11)
                        desenhar_camadas_kml(m_hull, geometrias_kml_dict, kml_laranja_dict)
                        try:
                            # Adicionar camada de improdutividade
                            if gdf_improdutividade is not None:
                                folium.Choropleth(
                                    geo_data=gdf_improdutividade,
                                    name='Improdutividade Histórica',
                                    data=gdf_improdutividade,
                                    columns=['h3_index', 'taxa_improdutividade_%'],
                                    key_on='feature.properties.h3_index',
                                    fill_color='YlOrRd',
                                    fill_opacity=0.6,
                                    line_opacity=0.2,
                                    legend_name='Taxa de Improdutividade Histórica (%)',
                                    highlight=True
                                ).add_to(m_hull)

                            hulls = gdf_clusters_reais.dissolve(by='cluster').convex_hull
                            gdf_hulls = gpd.GeoDataFrame(geometry=hulls).reset_index()
                            
                            counts = gdf_clusters_reais.groupby('cluster').size().rename('contagem')
                            gdf_hulls = gdf_hulls.merge(counts, on='cluster')
                            gdf_hulls_proj = gdf_hulls.to_crs("EPSG:3857")
                            gdf_hulls['area_km2'] = (gdf_hulls_proj.geometry.area / 1_000_000).round(2)
                            gdf_hulls['densidade'] = (gdf_hulls['contagem'] / gdf_hulls['area_km2']).replace([np.inf, -np.inf], 0).round(2)

                            # Análise de Improdutividade por Cluster
                            if gdf_improdutividade is not None:
                                intersecting_hexagons = gpd.sjoin(gdf_hulls, gdf_improdutividade, how="inner", predicate="intersects")
                                aggregated_stats = intersecting_hexagons.groupby('cluster').agg(
                                    taxa_improdutividade_%_media=('taxa_improdutividade_%', 'mean'),
                                    total_servicos_sum=('total_servicos', 'sum'),
                                    menor_data_min=('menor_data', 'min'),
                                    maior_data_max=('maior_data', 'max'),
                                    top_motivos_list=('top_motivos', 'sum')
                                )
                                
                                def aggregate_top_motives(motives_list):
                                    flat_list = [tuple(d.items()) for sublist in motives_list for d in sublist]
                                    counts = Counter(item['motivo'] for sublist in motives_list for item in sublist)
                                    total_improdutivos = sum(counts.values())
                                    top_5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
                                    return [{'motivo': m, 'rep_%': (c / total_improdutivos * 100) if total_improdutivos > 0 else 0} for m, c in top_5]

                                aggregated_stats['top_motivos_agregados'] = aggregated_stats['top_motivos_list'].apply(aggregate_top_motives)
                                gdf_hulls = gdf_hulls.merge(aggregated_stats, on='cluster', how='left')
                            
                            gdf_hulls['tooltip'] = gdf_hulls.apply(lambda row: f"<b>Cluster ID: {row['cluster']}</b><br>Nº de Cortes: {row['contagem']}<br>Área (km²): {row['area_km2']}<br>Cortes/km²: {row['densidade']}" + formatar_tooltip_improdutividade(row), axis=1)

                            folium.GeoJson(
                                gdf_hulls, 
                                style_function=lambda x: {'color': 'blue', 'weight': 2.5, 'fillColor': 'blue', 'fillOpacity': 0.2}, 
                                tooltip=folium.GeoJsonTooltip(fields=['tooltip'], aliases=[''], localize=True, sticky=True, style="white-space: pre-wrap;")
                            ).add_to(m_hull)
                            st_folium(m_hull, use_container_width=True, height=700)
                        except Exception as e: st.warning(f"Não foi possível desenhar os contornos. Erro: {e}")
                    else: st.warning("Nenhum cluster para desenhar.")
            tab_index += 1

            if df_metas is not None:
                with tabs[tab_index]: # Pacotes de Trabalho
                    # ... (código da aba com a nova camada de improdutividade)
                tab_index += 1
            
            with tabs[tab_index]: # Painel de Risco Climático
                # ... (código da aba sem alterações)
            tab_index += 1

            with tabs[tab_index]: # Metodologia
                # ... (código da aba sem alterações)
        else: st.warning("Nenhum dado para exibir com os filtros atuais.")
else: st.info("Aguardando o upload de um arquivo para iniciar a análise.")
