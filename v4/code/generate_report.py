#!/usr/bin/env python3
"""
generate_report.py
Genera un reporte visual HTML con estadísticas y visualizaciones de los resultados

Uso:
    python generate_report.py --csv results.csv --images test_images/ --out report.html
"""
import pandas as pd
import argparse
import os
from collections import Counter

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Reporte de Reconocimiento de Cartas</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        .stat-card {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-value {{
            font-size: 3em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 1.1em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .section {{
            padding: 40px;
        }}
        .section h2 {{
            font-size: 2em;
            margin-bottom: 20px;
            color: #1e3c72;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .table-container {{
            overflow-x: auto;
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}
        th {{
            background: #1e3c72;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .status-ok {{
            color: #28a745;
            font-weight: bold;
        }}
        .status-low {{
            color: #ffc107;
            font-weight: bold;
        }}
        .status-mismatch {{
            color: #dc3545;
            font-weight: bold;
        }}
        .score-high {{
            color: #28a745;
        }}
        .score-medium {{
            color: #ffc107;
        }}
        .score-low {{
            color: #dc3545;
        }}
        .distribution {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .dist-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}
        .dist-label {{
            font-size: 1.2em;
            font-weight: bold;
            color: #1e3c72;
            margin-bottom: 10px;
        }}
        .dist-value {{
            font-size: 2em;
            color: #667eea;
        }}
        .footer {{
            background: #1e3c72;
            color: white;
            text-align: center;
            padding: 20px;
            margin-top: 40px;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .badge-success {{
            background: #28a745;
            color: white;
        }}
        .badge-warning {{
            background: #ffc107;
            color: #333;
        }}
        .badge-danger {{
            background: #dc3545;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🃏 Reporte de Reconocimiento de Cartas</h1>
            <p>Sistema de Visión Artificial - Técnicas Clásicas</p>
        </div>
        
        <div class="stats">
            {stats_cards}
        </div>
        
        <div class="section">
            <h2>📊 Distribución de Detecciones</h2>
            <div class="distribution">
                {distribution_cards}
            </div>
        </div>
        
        <div class="section">
            <h2>📋 Detalle de Detecciones</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Imagen</th>
                            <th>ID</th>
                            <th>Valor</th>
                            <th>Score Valor</th>
                            <th>Palo</th>
                            <th>Score Palo</th>
                            <th>Estado</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="footer">
            <p>Generado automáticamente por generate_report.py</p>
            <p>Sistema de Reconocimiento de Cartas - Visión Artificial 2025</p>
        </div>
    </div>
</body>
</html>
"""

def generate_stats_cards(df):
    """Genera tarjetas de estadísticas"""
    total = len(df)
    ok = len(df[df['status'] == 'ok'])
    images = df['filename'].nunique()
    avg_val = df['value_score'].mean()
    avg_suit = df['suit_score'].mean()
    
    cards = f"""
        <div class="stat-card">
            <div class="stat-label">Total Detecciones</div>
            <div class="stat-value">{total}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Detecciones OK</div>
            <div class="stat-value">{ok}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Imágenes Procesadas</div>
            <div class="stat-value">{images}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Tasa de Éxito</div>
            <div class="stat-value">{ok/total*100:.1f}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Confianza Valor</div>
            <div class="stat-value">{avg_val:.2f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Confianza Palo</div>
            <div class="stat-value">{avg_suit:.2f}</div>
        </div>
    """
    return cards

def generate_distribution_cards(df):
    """Genera tarjetas de distribución"""
    status_count = df['status'].value_counts()
    value_count = df['value'].value_counts()
    suit_count = df['suit'].value_counts()
    
    cards = "<div class='dist-card'><div class='dist-label'>Estados</div>"
    for status, count in status_count.items():
        cards += f"<div>{status}: <span class='dist-value'>{count}</span></div>"
    cards += "</div>"
    
    cards += "<div class='dist-card'><div class='dist-label'>Valores Detectados</div>"
    for val, count in value_count.most_common(5):
        cards += f"<div>{val}: <span class='dist-value'>{count}</span></div>"
    cards += "</div>"
    
    cards += "<div class='dist-card'><div class='dist-label'>Palos Detectados</div>"
    for suit, count in suit_count.items():
        cards += f"<div>{suit}: <span class='dist-value'>{count}</span></div>"
    cards += "</div>"
    
    return cards

def generate_table_rows(df):
    """Genera filas de la tabla"""
    rows = ""
    for _, row in df.iterrows():
        # Formato de scores con colores
        val_score = row['value_score']
        suit_score = row['suit_score']
        
        val_class = 'score-high' if val_score > 0.7 else 'score-medium' if val_score > 0.4 else 'score-low'
        suit_class = 'score-high' if suit_score > 0.7 else 'score-medium' if suit_score > 0.4 else 'score-low'
        
        # Estado con badge
        status = row['status']
        if status == 'ok':
            status_html = "<span class='badge badge-success'>OK</span>"
        elif status == 'low_confidence':
            status_html = "<span class='badge badge-warning'>Baja Confianza</span>"
        else:
            status_html = "<span class='badge badge-danger'>Pip Mismatch</span>"
        
        rows += f"""
        <tr>
            <td>{row['filename']}</td>
            <td>{row['card_id']}</td>
            <td><strong>{row['value']}</strong></td>
            <td class='{val_class}'>{val_score:.3f}</td>
            <td><strong>{row['suit']}</strong></td>
            <td class='{suit_class}'>{suit_score:.3f}</td>
            <td>{status_html}</td>
        </tr>
        """
    return rows

def main():
    parser = argparse.ArgumentParser(description="Genera reporte HTML de resultados")
    parser.add_argument("--csv", required=True, help="Archivo CSV con resultados")
    parser.add_argument("--out", default="report.html", help="Archivo HTML de salida")
    args = parser.parse_args()
    
    # Cargar datos
    if not os.path.exists(args.csv):
        print(f"ERROR: No se encontró {args.csv}")
        return
    
    df = pd.read_csv(args.csv)
    
    if len(df) == 0:
        print("ERROR: CSV vacío")
        return
    
    print(f"Generando reporte de {len(df)} detecciones...")
    
    # Generar componentes
    stats_cards = generate_stats_cards(df)
    distribution_cards = generate_distribution_cards(df)
    table_rows = generate_table_rows(df)
    
    # Generar HTML
    html = HTML_TEMPLATE.format(
        stats_cards=stats_cards,
        distribution_cards=distribution_cards,
        table_rows=table_rows
    )
    
    # Guardar
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Reporte generado: {args.out}")
    print(f"  Total detecciones: {len(df)}")
    print(f"  Detecciones OK: {len(df[df['status'] == 'ok'])}")
    print(f"  Tasa de éxito: {len(df[df['status'] == 'ok'])/len(df)*100:.1f}%")

if __name__ == "__main__":
    main()