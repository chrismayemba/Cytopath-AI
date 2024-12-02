import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Navigation from './components/Navigation';
import ImageUpload from './components/ImageUpload';
import Analysis from './components/Analysis';
import Report from './components/Report';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#f50057',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Navigation />
        <Routes>
          <Route path="/" element={<ImageUpload />} />
          <Route path="/analysis/:imageId" element={<Analysis />} />
          <Route path="/report/:analysisId" element={<Report />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
