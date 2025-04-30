# reports.py
from io import BytesIO, StringIO
import csv
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from flask import send_file
from database import get_pg_connection
from datetime import datetime

def generate_pdf():
    """Generate PDF report of suspicious transactions"""
    try:
        conn = get_pg_connection()
        if not conn:
            return {'error': 'Database connection failed'}, 500
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT transaction_id, customer_id, transaction_time, amount, 
                   transaction_type, transaction_description
            FROM monitored_transactions 
            WHERE is_suspicious = true 
            ORDER BY transaction_time DESC
        """)
        
        suspicious_transactions = cursor.fetchall()
        
        if not suspicious_transactions:
            cursor.close()
            conn.close()
            return {'message': 'No suspicious transactions found'}, 404
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        data = [['Transaction ID', 'Customer ID', 'Time', 'Amount', 'Type', 'Description']]
        for transaction in suspicious_transactions:
            data.append([
                str(transaction[0]), str(transaction[1]), str(transaction[2]),
                f"${transaction[3]:,.2f}", transaction[4], transaction[5]
            ])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(table)
        doc.build(elements)
        
        buffer.seek(0)
        cursor.close()
        conn.close()
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'suspicious_transactions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
        
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
        return {'error': str(e)}, 500

def generate_csv():
    """Generate CSV report of suspicious transactions"""
    try:
        conn = get_pg_connection()
        if not conn:
            return {'error': 'Database connection failed'}, 500
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT transaction_id, customer_id, transaction_time, amount, 
                   transaction_type, transaction_description
            FROM monitored_transactions 
            WHERE is_suspicious = true 
            ORDER BY transaction_time DESC
        """)
        
        suspicious_transactions = cursor.fetchall()
        
        if not suspicious_transactions:
            cursor.close()
            conn.close()
            return {'message': 'No suspicious transactions found'}, 404
        
        string_buffer = StringIO()
        writer = csv.writer(string_buffer)
        writer.writerow(['Transaction ID', 'Customer ID', 'Time', 'Amount', 'Type', 'Description'])
        
        for transaction in suspicious_transactions:
            writer.writerow([
                str(transaction[0]), str(transaction[1]), str(transaction[2]),
                f"${transaction[3]:,.2f}", transaction[4], transaction[5]
            ])
        
        output = BytesIO()
        output.write(string_buffer.getvalue().encode('utf-8'))
        string_buffer.close()
        
        output.seek(0)
        cursor.close()
        conn.close()
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'suspicious_transactions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
    except Exception as e:
        print(f"Error generating CSV report: {str(e)}")
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
        return {'error': str(e)}, 500