import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Container,
  Paper,
  Typography,
  LinearProgress,
  Grid,
  Chip,
} from '@mui/material';
import axios from 'axios';

interface AnalysisResult {
  classification: string;
  confidence: number;
  regions_of_interest: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
  }>;
}

const Analysis: React.FC = () => {
  const { imageId } = useParams<{ imageId: string }>();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        const response = await axios.post('http://localhost:8000/analyze/', {
          file_path: imageId,
        });
        setResult(response.data);
      } catch (error) {
        console.error('Analysis failed:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalysis();
  }, [imageId]);

  const getBethesdaColor = (classification: string) => {
    const colors = {
      'NILM': 'success',
      'LSIL': 'warning',
      'HSIL': 'error',
      'Squamous Cell Carcinoma': 'error',
      'Other Abnormalities': 'warning',
    } as const;
    
    return colors[classification as keyof typeof colors] || 'default';
  };

  if (loading) {
    return (
      <Container maxWidth="md" sx={{ mt: 4 }}>
        <LinearProgress />
        <Typography sx={{ mt: 2 }} align="center">
          Analyzing image...
        </Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" gutterBottom align="center">
          Analysis Results
        </Typography>

        {result && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
                <Chip
                  label={result.classification}
                  color={getBethesdaColor(result.classification)}
                  size="large"
                />
                <Chip
                  label={`Confidence: ${(result.confidence * 100).toFixed(1)}%`}
                  variant="outlined"
                />
              </Box>
            </Grid>

            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Detected Regions of Interest
              </Typography>
              {/* TODO: Add canvas overlay to show regions of interest */}
            </Grid>
          </Grid>
        )}
      </Paper>
    </Container>
  );
};

export default Analysis;
