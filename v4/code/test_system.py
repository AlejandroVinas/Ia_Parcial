#!/usr/bin/env python3
"""
test_system.py
Script de pruebas automatizadas para validar el sistema de reconocimiento

Uso:
    python test_system.py --images test_images/ --templates templates/ --ground_truth ground_truth.csv

Ground truth format (CSV):
    filename,expected_cards
    test001.jpg,"A-hearts"
    test002.jpg,"K-spades,Q-diamonds"
"""
import argparse
import pandas as pd
import os
import subprocess
import sys

def parse_expected_cards(cards_str):
    """
    Parse expected cards string
    Format: "VALUE-SUIT,VALUE-SUIT,..."
    Example: "A-hearts,K-spades"
    """
    if pd.isna(cards_str) or cards_str.strip() == "":
        return []
    cards = []
    for card in cards_str.split(','):
        card = card.strip()
        if '-' in card:
            value, suit = card.split('-')
            cards.append((value.strip(), suit.strip()))
    return cards

def evaluate_detection(detected_df, expected_cards):
    """
    Evaluate detection results against expected cards
    
    Returns:
        (correct, total, details)
    """
    detected_cards = []
    for _, row in detected_df.iterrows():
        if row['status'] == 'ok':
            detected_cards.append((row['value'], row['suit']))
    
    # Count matches
    correct = 0
    matched_expected = []
    matched_detected = []
    
    for exp_card in expected_cards:
        if exp_card in detected_cards and exp_card not in matched_expected:
            correct += 1
            matched_expected.append(exp_card)
            matched_detected.append(exp_card)
    
    total = len(expected_cards)
    
    # Detailed results
    details = {
        'expected': expected_cards,
        'detected': detected_cards,
        'matched': matched_expected,
        'missing': [c for c in expected_cards if c not in matched_expected],
        'extra': [c for c in detected_cards if c not in matched_detected]
    }
    
    return correct, total, details

def run_detection(image_path, templates_dir):
    """
    Run detection on single image and return results
    """
    import tempfile
    import csv
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_csv = f.name
    
    try:
        # Run detection script
        cmd = [
            sys.executable,
            'detect_and_recognize.py',
            '--input', image_path,
            '--templates', templates_dir,
            '--out', temp_csv,
            '--no-debug'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERROR running detection: {result.stderr}")
            return None
        
        # Read results
        if os.path.exists(temp_csv):
            df = pd.read_csv(temp_csv)
            return df
        else:
            return None
            
    finally:
        # Cleanup
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

def print_test_results(filename, correct, total, details):
    """Print formatted test results"""
    if total == 0:
        status = "⚠️  NO EXPECTED"
        color = "\033[93m"  # Yellow
    elif correct == total:
        status = "✓ PASS"
        color = "\033[92m"  # Green
    else:
        status = "✗ FAIL"
        color = "\033[91m"  # Red
    
    reset = "\033[0m"
    
    print(f"\n{color}{status}{reset} {filename} ({correct}/{total})")
    
    if details['expected']:
        exp_str = ', '.join([f"{v}-{s}" for v,s in details['expected']])
        print(f"  Expected: {exp_str}")
    
    if details['detected']:
        det_str = ', '.join([f"{v}-{s}" for v,s in details['detected']])
        print(f"  Detected: {det_str}")
    
    if details['missing']:
        miss_str = ', '.join([f"{v}-{s}" for v,s in details['missing']])
        print(f"  ❌ Missing: {miss_str}")
    
    if details['extra']:
        extra_str = ', '.join([f"{v}-{s}" for v,s in details['extra']])
        print(f"  ➕ Extra: {extra_str}")

def main():
    parser = argparse.ArgumentParser(
        description="Test automatizado del sistema de reconocimiento"
    )
    parser.add_argument("--images", required=True, help="Carpeta con imágenes de test")
    parser.add_argument("--templates", required=True, help="Carpeta de plantillas")
    parser.add_argument("--ground_truth", required=True, 
                       help="CSV con ground truth (filename, expected_cards)")
    args = parser.parse_args()
    
    # Load ground truth
    if not os.path.exists(args.ground_truth):
        print(f"ERROR: Ground truth file not found: {args.ground_truth}")
        return
    
    gt_df = pd.read_csv(args.ground_truth)
    
    if 'filename' not in gt_df.columns or 'expected_cards' not in gt_df.columns:
        print("ERROR: Ground truth CSV must have 'filename' and 'expected_cards' columns")
        return
    
    print("=" * 70)
    print("SISTEMA DE PRUEBAS AUTOMATIZADO")
    print("=" * 70)
    print(f"Templates: {args.templates}")
    print(f"Test images: {len(gt_df)}")
    print("=" * 70)
    
    # Run tests
    total_correct = 0
    total_expected = 0
    test_results = []
    
    for idx, row in gt_df.iterrows():
        filename = row['filename']
        expected_str = row['expected_cards']
        
        image_path = os.path.join(args.images, filename)
        
        if not os.path.exists(image_path):
            print(f"\n⚠️  SKIP {filename} (file not found)")
            continue
        
        expected_cards = parse_expected_cards(expected_str)
        
        # Run detection
        detected_df = run_detection(image_path, args.templates)
        
        if detected_df is None:
            print(f"\n❌ ERROR {filename} (detection failed)")
            continue
        
        # Evaluate
        correct, total, details = evaluate_detection(detected_df, expected_cards)
        
        total_correct += correct
        total_expected += total
        
        # Print results
        print_test_results(filename, correct, total, details)
        
        test_results.append({
            'filename': filename,
            'correct': correct,
            'total': total,
            'accuracy': correct / total if total > 0 else 0
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Total detecciones correctas: {total_correct}/{total_expected}")
    
    if total_expected > 0:
        accuracy = (total_correct / total_expected) * 100
        print(f"Precisión global: {accuracy:.1f}%")
        
        if accuracy >= 90:
            print("🏆 EXCELENTE - Sistema funciona correctamente")
        elif accuracy >= 70:
            print("✓ BUENO - Sistema funcional con margen de mejora")
        elif accuracy >= 50:
            print("⚠️  REGULAR - Se requieren ajustes")
        else:
            print("❌ INSUFICIENTE - Revisar configuración y plantillas")
    
    # Per-image summary
    if test_results:
        print("\nResultados por imagen:")
        for result in test_results:
            acc = result['accuracy'] * 100
            status = "✓" if acc == 100 else "✗"
            print(f"  {status} {result['filename']}: {result['correct']}/{result['total']} ({acc:.0f}%)")
    
    print("=" * 70)

if __name__ == "__main__":
    main()