package com.npl;

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
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

public class DocumentTextAnalysis {

    // Rutas y configuraciones
    private static final String CATEGORIAS_PATH = "categorias.txt"; // Archivo con categorías y palabras clave
    private static final int UMBRAL_LONGITUD_PALABRA = 3; // Mínima longitud para considerar una palabra como clave
    private static StanfordCoreNLP pipeline; // Objeto de procesamiento de Stanford CoreNLP
    private static Map<String, TrieNode> categoriasTries = new HashMap<>(); // Tries para clasificar palabras clave por categorías

    public static void main(String[] args) throws Exception {
        // Inicializa el pipeline de Stanford CoreNLP con anotadores para análisis textual
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner");
        props.setProperty("tokenize.language", "es");
        pipeline = new StanfordCoreNLP(props);

        // Carga las categorías desde el archivo `categorias.txt` en una estructura Trie
        cargarCategoriasEnTries(CATEGORIAS_PATH);

        // Lee un archivo de texto o PDF y obtiene su contenido como String
        String texto = leerArchivo("documento.txt");

        // Procesa el texto para identificar entidades y palabras clave
        procesarDocumento(texto);
    }

    public static void procesarDocumento(String texto) {
        Annotation documento = new Annotation(texto);
        pipeline.annotate(documento); // Anota el texto para análisis lingüístico

        // Identificación del autor utilizando entidades nombradas (NER)
        StringBuilder autorBuilder = new StringBuilder();
        boolean isPerson = false; // Bandera para concatenar nombres etiquetados como `PERSON`
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
                    break; // Detenemos la concatenación al salir del bloque de `PERSON`
                }
            }
        }
        String autor = autorBuilder.toString().trim();

        // Si no se encuentra un autor con CoreNLP, se usa una expresión regular como respaldo
        if (autor.isEmpty()) {
            Pattern nombrePattern = Pattern.compile("\\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\\s[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\\b");
            Matcher matcher = nombrePattern.matcher(texto);
            if (matcher.find()) {
                autor = matcher.group(1);
            }
        }

        // Identificación de la fecha mediante etiquetas `DATE` o expresiones regulares
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

        // Clasificación de palabras clave en categorías utilizando los Tries
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

        // Muestra los resultados obtenidos
        if (autor.isEmpty()) {
            System.out.println("Autor: No identificado");
        } else {
            System.out.println("Autor: " + autor);
        }

        if (fecha.isEmpty()) {
            System.out.println("Fecha: No identificada");
        } else {
            System.out.println("Fecha: " + fecha);
        }

        System.out.println("Palabras clave por categoría:");
        for (Map.Entry<String, Set<String>> entry : palabrasClavePorCategoria.entrySet()) {
            if (!entry.getValue().isEmpty()) {
                System.out.println(entry.getKey() + ": " + entry.getValue());
            }
        }
    }

    private static void cargarCategoriasEnTries(String path) {
        try (InputStream is = DocumentTextAnalysis.class.getClassLoader().getResourceAsStream(path);
             Scanner scanner = new Scanner(is)) {
            String categoriaActual = null;
            while (scanner.hasNextLine()) {
                String linea = scanner.nextLine().trim();
                if (linea.isEmpty()) {
                    continue;
                }
                if (linea.startsWith("#")) { // Nueva categoría
                    categoriaActual = linea.substring(1).trim();
                    categoriasTries.put(categoriaActual, new TrieNode());
                } else if (categoriaActual != null) {
                    insertarEnTrie(categoriasTries.get(categoriaActual), linea.toLowerCase());
                }
            }
        } catch (Exception e) {
            System.err.println("Error al cargar las categorías: " + e.getMessage());
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
        InputStream is = DocumentTextAnalysis.class.getClassLoader().getResourceAsStream(rutaArchivo);
        if (is != null) {
            return new String(is.readAllBytes(), "UTF-8");
        } else {
            throw new IllegalArgumentException("Archivo no encontrado en recursos: " + rutaArchivo);
        }
    }
}

