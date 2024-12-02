from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.pdfgen import canvas
from datetime import datetime
import cv2
import numpy as np
from pathlib import Path
import io
from typing import Dict, List, Optional
from ..database.config import get_postgres_connection, get_mongo_client

class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self.mongo_db = get_mongo_client()
        
        # Add custom styles
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        ))
        
    def _create_header(self, doc: SimpleDocTemplate, canvas: canvas, *args) -> None:
        """Create PDF header"""
        canvas.saveState()
        canvas.setFont('Helvetica', 10)
        canvas.drawString(doc.leftMargin, doc.height + doc.topMargin + 10,
                         f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        canvas.drawString(doc.width + doc.rightMargin - 2*inch, doc.height + doc.topMargin + 10,
                         "Cervical Lesion Detection Report")
        canvas.restoreState()
    
    def _create_summary_table(self, analysis_data: Dict) -> Table:
        """Create summary table with analysis results"""
        data = [
            ['Parameter', 'Value'],
            ['Analysis Date', analysis_data['analysis_date'].strftime('%Y-%m-%d %H:%M:%S')],
            ['Classification', analysis_data['classification']],
            ['Confidence', f"{analysis_data['confidence']*100:.1f}%"],
            ['Bethesda Category', analysis_data['bethesda_category']],
            ['Number of Cells Detected', str(analysis_data['num_cells_detected'])]
        ]
        
        if analysis_data.get('validated_by_pathologist'):
            data.extend([
                ['Validation Date', analysis_data['validation_date'].strftime('%Y-%m-%d %H:%M:%S')],
                ['Pathologist Classification', analysis_data['validated_classification']],
                ['Pathologist Notes', analysis_data.get('pathologist_notes', 'N/A')]
            ])
        
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        return table
    
    def _prepare_image_with_annotations(self, 
                                      image_path: str,
                                      cell_regions: List[Dict]) -> Image:
        """Prepare image with annotated cell regions"""
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw cell regions
        for region in cell_regions:
            coords = region['coordinates']
            x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
            
            # Color based on classification
            if region.get('pathologist_annotation'):
                color = (0, 255, 0)  # Green for validated
            else:
                confidence = region.get('confidence', 0)
                color = (
                    int(255 * (1 - confidence)),  # Red component
                    int(255 * confidence),        # Green component
                    0                             # Blue component
                )
            
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        # Convert to reportlab Image
        img_byte_arr = io.BytesIO()
        cv2.imwrite(img_byte_arr, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        img_byte_arr = img_byte_arr.getvalue()
        
        return Image(img_byte_arr)
    
    def generate_report(self, analysis_id: str) -> str:
        """Generate PDF report for analysis results"""
        # Get analysis data
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ar.*, im.filename, im.upload_date,
                           im.staining_method, im.microscope_info
                    FROM analysis_results ar
                    JOIN image_metadata im ON ar.image_id = im.id
                    WHERE ar.id = %s
                """, (analysis_id,))
                analysis_data = dict(cur.fetchone())
        
        # Get cell regions
        cell_regions = list(self.mongo_db.cell_regions.find(
            {"analysis_id": analysis_id}
        ))
        
        # Create PDF
        output_path = self.output_dir / f"report_{analysis_id}.pdf"
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build content
        content = []
        
        # Title
        title = Paragraph("Cervical Lesion Analysis Report", self.styles['CustomTitle'])
        content.append(title)
        content.append(Spacer(1, 12))
        
        # Summary table
        content.append(self._create_summary_table(analysis_data))
        content.append(Spacer(1, 20))
        
        # Image with annotations
        if cell_regions:
            content.append(Paragraph("Analyzed Image with Detected Cells", self.styles['Heading2']))
            content.append(Spacer(1, 12))
            content.append(self._prepare_image_with_annotations(
                analysis_data['filename'],
                cell_regions
            ))
            content.append(Spacer(1, 20))
        
        # Cell details
        if cell_regions:
            content.append(Paragraph("Detected Cell Regions", self.styles['Heading2']))
            content.append(Spacer(1, 12))
            
            cell_data = [['Region ID', 'Classification', 'Confidence', 'Validated']]
            for region in cell_regions:
                cell_data.append([
                    region['region_id'],
                    region.get('classification', 'N/A'),
                    f"{region.get('confidence', 0)*100:.1f}%",
                    'Yes' if region.get('pathologist_annotation') else 'No'
                ])
            
            cell_table = Table(cell_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 1*inch])
            cell_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            content.append(cell_table)
        
        # Build PDF
        doc.build(content, onFirstPage=self._create_header, onLaterPages=self._create_header)
        
        return str(output_path)
