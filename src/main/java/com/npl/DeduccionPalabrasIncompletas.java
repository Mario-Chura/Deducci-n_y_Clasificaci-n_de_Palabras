package com.npl;

import java.io.InputStream;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Scanner;
import java.util.Set;
import java.util.regex.Pattern;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class DeduccionPalabrasIncompletas {

    // Ruta del archivo de diccionario
    private static final String DICCIONARIO_PATH = "dic.txt";
    // Umbral mínimo para evaluar palabras (evita procesar palabras muy cortas)
    private static final int UMBRAL_LONGITUD_PALABRA = 3;
    // Umbral máximo de distancia de Levenshtein para considerar palabras similares
    private static final int UMBRAL_DISTANCIA = 2;

    // Instancia estática del pipeline para procesamiento NLP
    private static StanfordCoreNLP pipeline;

    public static void main(String[] args) {

        // 1. Inicializar Stanford CoreNLP con propiedades para idioma español
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos"); // Tokenización y etiquetado gramatical
        props.setProperty("tokenize.language", "es"); // Configurar idioma español
        props.setProperty("pos.model", "edu/stanford/nlp/models/pos-tagger/spanish-ud.tagger"); // Modelo POS

        // Crear pipeline NLP
        pipeline = new StanfordCoreNLP(props);

        // 2. Cargar diccionario desde el archivo especificado
        Set<String> diccionario = cargarDiccionario(DICCIONARIO_PATH);
        // Ejemplo de texto histórico (no se utiliza posteriormente)
        String textoHistorico = "La ciuda de Arequip fundda en 1540.";

        // 3. Evaluar correcciones en frases predefinidas
        long startTime = System.nanoTime(); // Medir tiempo de inicio
        evaluarCorrecciones(diccionario);
        long endTime = System.nanoTime(); // Medir tiempo de finalización
        long duration = (endTime - startTime);  // Calcular duración en nanosegundos

        // Mostrar el tiempo de ejecución
        System.out.println("\nTiempo de ejecución: " + duration + " nanosegundos");
    }

    // Método para evaluar y corregir errores ortográficos en frases
    public static void evaluarCorrecciones(Set<String> diccionario) {
        // Lista de frases con errores ortográficos predefinidos
        String[] frasesIncorrectas = {
                "El trbajo eta inompleto poqe fue apido",
                "La combnicación de cocolates y flores fué ideal",
                "Me gustaría comprar un perro de raca pequeña",
                "Estoy deseando probar esa nuva marca de helado",
                "No entendí el acertijo, era demaciado complicado",
                "Mi hermano compitió en una competición de artes marciales",
                "El café estava delicioso, no me gusto el pastel",
                "El estudiante tenía que entregar un travajo muy extenso",
                "Fuimos al cine pero la película estubo aburrida",
                "Esa novela tiene una trama apasionante, la recomiendo"
        };

        // Iterar sobre cada frase para procesarla
        for (String frase : frasesIncorrectas) {
            System.out.println("\nFrase original: " + frase);
            // Anotar la frase usando Stanford CoreNLP
            Annotation documento = new Annotation(frase);
            pipeline.annotate(documento);

            // Procesar cada oración en la frase
            for (CoreMap oracion : documento.get(SentencesAnnotation.class)) {
                for (CoreLabel token : oracion.get(TokensAnnotation.class)) {
                    // Obtener la palabra original
                    String palabraOriginal = token.originalText();
                    // Corregir la palabra usando el diccionario
                    String palabraCorregida = corregirPalabra(palabraOriginal, diccionario);
                    // Mostrar la palabra corregida
                    System.out.println(palabraOriginal + " -> " + palabraCorregida);
                }
            }
        }
    }

    // Método para cargar el diccionario desde un archivo externo
    private static Set<String> cargarDiccionario(String path) {
        Set<String> palabras = new HashSet<>(); // Usar un conjunto para evitar duplicados
        try (InputStream is = DeduccionPalabrasIncompletas.class.getClassLoader().getResourceAsStream(path);
                Scanner scanner = new Scanner(is)) {
            while (scanner.hasNext()) {
                palabras.add(scanner.nextLine().toLowerCase()); // Convertir palabras a minúsculas
            }
        } catch (Exception e) {
            // Mostrar mensaje de error si falla la carga del diccionario
            System.err.println("Error al cargar el diccionario: " + e.getMessage());
        }
        return palabras;
    }

    // Función para corregir una palabra basándose en la distancia de Levenshtein
    private static String corregirPalabra(String palabra, Set<String> diccionario) {
        String mejorCandidata = palabra;
        int menorDistancia = Integer.MAX_VALUE;

        // Comparar cada palabra del diccionario para encontrar la más cercana
        for (String candidata : diccionario) {
            int distancia = calcularDistanciaLevenshtein(palabra.toLowerCase(), candidata);
            if (distancia < menorDistancia && distancia <= UMBRAL_DISTANCIA) {
                menorDistancia = distancia;
                mejorCandidata = candidata; // Actualizar mejor candidata
            }
        }
        return mejorCandidata; // Devolver la mejor coincidencia encontrada
    }

    // Implementación del algoritmo de distancia de Levenshtein para medir similitud
    private static int calcularDistanciaLevenshtein(String a, String b) {
        int[][] dp = new int[a.length() + 1][b.length() + 1];

        for (int i = 0; i <= a.length(); i++) {
            for (int j = 0; j <= b.length(); j++) {
                if (i == 0) {
                    dp[i][j] = j; // Coste de inserción
                } else if (j == 0) {
                    dp[i][j] = i; // Coste de eliminación
                } else {
                    dp[i][j] = Math.min(
                            Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1), // Eliminación o inserción
                            dp[i - 1][j - 1] + (a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1)); // Sustitución
                }
            }
        }

        return dp[a.length()][b.length()]; // Devolver la distancia calculada
    }

    // Método para verificar si una palabra es numérica o puntuación
    private static boolean esNumericaOPuntuacion(String palabra) {
        return palabra.chars().allMatch(Character::isDigit) || Pattern.matches("\\p{Punct}", palabra);
    }
}

