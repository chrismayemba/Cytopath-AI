import React from 'react';
import {
  Container,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
} from '@mui/material';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';

interface ReportProps {
  analysisId?: string;
}

const Report: React.FC<ReportProps> = ({ analysisId }) => {
  const handleDownloadPDF = () => {
    // TODO: Implement PDF download functionality
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" gutterBottom align="center">
          Analysis Report
        </Typography>

        <TableContainer component={Paper} sx={{ mt: 4 }}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Parameter</TableCell>
                <TableCell>Value</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              <TableRow>
                <TableCell>Analysis Date</TableCell>
                <TableCell>{new Date().toLocaleDateString()}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Classification</TableCell>
                <TableCell>NILM</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Confidence Score</TableCell>
                <TableCell>95%</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Regions of Interest</TableCell>
                <TableCell>3</TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </TableContainer>

        <Button
          variant="contained"
          startIcon={<PictureAsPdfIcon />}
          onClick={handleDownloadPDF}
          sx={{ mt: 4 }}
        >
          Download PDF Report
        </Button>
      </Paper>
    </Container>
  );
};

export default Report;
