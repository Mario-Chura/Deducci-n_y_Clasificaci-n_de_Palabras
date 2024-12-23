package com.npl;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.InputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class DocumentAnalysisApp {
    private static final String CATEGORIAS_PATH = "categorias.txt";
    private static StanfordCoreNLP pipeline;
    private static Map<String, TrieNode> categoriasTries = new HashMap<>();

    public static void main(String[] args) throws Exception {
        // Inicializar el pipeline NLP
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner");
        props.setProperty("tokenize.language", "es");
        pipeline = new StanfordCoreNLP(props);

        // Cargar las categorías en diferentes tries
        cargarCategoriasEnTries(CATEGORIAS_PATH);

        // Crear la interfaz gráfica
        JFrame frame = new JFrame("Análisis de Documentos");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(600, 400);

        JPanel panel = new JPanel();
        panel.setLayout(new BorderLayout());

        JTextArea resultArea = new JTextArea();
        resultArea.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(resultArea);

        JButton loadButton = new JButton("Cargar Documento");
        loadButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JFileChooser fileChooser = new JFileChooser();
                int option = fileChooser.showOpenDialog(frame);
                if (option == JFileChooser.APPROVE_OPTION) {
                    File selectedFile = fileChooser.getSelectedFile();
                    try {
                        String texto = leerArchivo(selectedFile.getAbsolutePath());
                        String results = procesarDocumento(texto);
                        resultArea.setText(results);
                    } catch (Exception ex) {
                        JOptionPane.showMessageDialog(frame, "Error al procesar el documento: " + ex.getMessage());
                    }
                }
            }
        });

        panel.add(loadButton, BorderLayout.NORTH);
        panel.add(scrollPane, BorderLayout.CENTER);

        frame.add(panel);
        frame.setVisible(true);
    }

    private static String procesarDocumento(String texto) {
        Annotation documento = new Annotation(texto);
        pipeline.annotate(documento);

        StringBuilder autorBuilder = new StringBuilder();
        boolean isPerson = false;

        for (CoreMap oracion : documento.get(SentencesAnnotation.class)) {
            for (CoreLabel token : oracion.get(TokensAnnotation.class)) {
                String ner = token.get(NamedEntityTagAnnotation.class);
                if ("PERSON".equals(ner)) {
                    if (autorBuilder.length() > 0) {
                        autorBuilder.append(" ");
                    }
                    autorBuilder.append(token.originalText());
                    isPerson = true;
                } else if (isPerson) {
                    break;
                }
            }
        }
        String autor = autorBuilder.toString().trim();

        if (autor.isEmpty()) {
            Pattern nombrePattern = Pattern.compile("\\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\\s[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\\b");
            Matcher matcher = nombrePattern.matcher(texto);
            if (matcher.find()) {
                autor = matcher.group(1);
            }
        }

        String fecha = "";
        for (CoreMap oracion : documento.get(SentencesAnnotation.class)) {
            for (CoreLabel token : oracion.get(TokensAnnotation.class)) {
                String ner = token.get(NamedEntityTagAnnotation.class);
                if ("DATE".equals(ner)) {
                    fecha = token.originalText();
                }
            }
        }

        if (fecha.isEmpty()) {
            Pattern datePattern = Pattern.compile("\\b(\\d{1,2} de [a-zA-Z]+ de \\d{4})\\b");
            Matcher matcher = datePattern.matcher(texto);
            if (matcher.find()) {
                fecha = matcher.group(1);
            }
        }

        Map<String, Set<String>> palabrasClavePorCategoria = new HashMap<>();
        for (String categoria : categoriasTries.keySet()) {
            palabrasClavePorCategoria.put(categoria, new HashSet<>());
        }

        for (CoreMap oracion : documento.get(SentencesAnnotation.class)) {
            for (CoreLabel token : oracion.get(TokensAnnotation.class)) {
                String palabraOriginal = token.originalText();
                for (Map.Entry<String, TrieNode> entry : categoriasTries.entrySet()) {
                    String categoria = entry.getKey();
                    TrieNode trie = entry.getValue();
                    if (buscarEnTrie(trie, palabraOriginal.toLowerCase())) {
                        palabrasClavePorCategoria.get(categoria).add(palabraOriginal.toLowerCase());
                    }
                }
            }
        }

        StringBuilder resultados = new StringBuilder();
        resultados.append("Autor: ").append(autor.isEmpty() ? "No identificado" : autor).append("\n");
        resultados.append("Fecha: ").append(fecha.isEmpty() ? "No identificada" : fecha).append("\n");
        resultados.append("Palabras clave por categoría:\n");
        for (Map.Entry<String, Set<String>> entry : palabrasClavePorCategoria.entrySet()) {
            if (!entry.getValue().isEmpty()) {
                resultados.append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
            }
        }

        return resultados.toString();
    }

    private static void cargarCategoriasEnTries(String path) throws Exception {
        InputStream is = DocumentAnalysisApp.class.getClassLoader().getResourceAsStream(path);
        if (is == null) {
            throw new Exception("Archivo de categorías no encontrado.");
        }
        Scanner scanner = new Scanner(is);
        String categoriaActual = null;
        while (scanner.hasNextLine()) {
            String linea = scanner.nextLine().trim();
            if (linea.isEmpty()) {
                continue;
            }
            if (linea.startsWith("#")) {
                categoriaActual = linea.substring(1).trim();
                categoriasTries.put(categoriaActual, new TrieNode());
            } else if (categoriaActual != null) {
                insertarEnTrie(categoriasTries.get(categoriaActual), linea.toLowerCase());
            }
        }
    }

    static class TrieNode {
        Map<Character, TrieNode> hijos = new HashMap<>();
        boolean esPalabra = false;
    }

    private static void insertarEnTrie(TrieNode nodo, String palabra) {
        for (char c : palabra.toCharArray()) {
            nodo.hijos.putIfAbsent(c, new TrieNode());
            nodo = nodo.hijos.get(c);
        }
        nodo.esPalabra = true;
    }

    private static boolean buscarEnTrie(TrieNode nodo, String palabra) {
        for (char c : palabra.toCharArray()) {
            nodo = nodo.hijos.get(c);
            if (nodo == null) {
                return false;
            }
        }
        return nodo.esPalabra;
    }

    private static String leerArchivo(String rutaArchivo) throws Exception {
        File file = new File(rutaArchivo);
        if (!file.exists()) {
            throw new Exception("Archivo no encontrado: " + rutaArchivo);
        }
        return new String(java.nio.file.Files.readAllBytes(file.toPath()), "UTF-8");
    }
}
