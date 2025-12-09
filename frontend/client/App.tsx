import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "@/contexts/ThemeContext";
import Index from "./pages/Index";
import Reports from "./components/Reports";
import NotFound from "./pages/NotFound";
import Layout from "./components/Layout";
import Forecast from "./components/Forecast";
import ModelEvaluation from "./components/ModelEvaluation";
import About from "./components/About";
import Dashboard from "./components/Dashboard";

const queryClient = new QueryClient();

export default function App() {
  return (
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            {/* <Routes>
              <Route element={<Layout />} />
              <Route path="/" element={<Index />} />
              <Route path="/reports" element={<Reports />} />
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
             
            
             <Routes>
              <Route element = {<Layout />}>
              
                <Route path="/" element={<Index />} />
                { <Route path="/forecast" element={<Forecast />} /> }
                <Route path="/model_evaluation" element={<ModelEvaluation />} />
                <Route path="/reports" element={<Reports />} />
                <Route path="/" element={<Dashboard />} />
                <Route path="/about" element={<About />} />

              </Route>
              
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Routes>
             
          </BrowserRouter>
        </TooltipProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
}
