import java.util.Scanner;

public class Main {
    public static boolean compareLe(int a, int b) {
        return a <= b;
    }
    public static boolean compareLe(double a, double b) {
        return a <= b;
    }
    public static boolean compareLe(String a, String b) {
        return a.compareTo(b) <= 0;
    }

    public static boolean compareLt(int a, int b) {
        return a < b;
    }
    public static boolean compareLt(double a, double b) {
        return a < b;
    }
    public static boolean compareLt(String a, String b) {
        return a.compareTo(b) < 0;
    }

    public static boolean compareGe(int a, int b) {
        return a >= b;
    }
    public static boolean compareGe(double a, double b) {
        return a >= b;
    }
    public static boolean compareGe(String a, String b) {
        return a.compareTo(b) >= 0;
    }

    public static boolean compareGt(int a, int b) {
        return a > b;
    }
    public static boolean compareGt(double a, double b) {
        return a > b;
    }
    public static boolean compareGt(String a, String b) {
        return a.compareTo(b) > 0;
    }

    public static boolean compareNe(int a, int b) {
        return a != b;
    }
    public static boolean compareNe(double a, double b) {
        return a != b;
    }
    public static boolean compareNe(String a, String b) {
        return !a.equals(b);
    }
    public static boolean compareNe(boolean a, boolean b) {
        return a != b;
    }

    public static boolean compareEq(int a, int b) {
        return a == b;
    }
    public static boolean compareEq(double a, double b) {
        return a == b;
    }
    public static boolean compareEq(String a, String b) {
        return a.equals(b);
    }
    public static boolean compareEq(boolean a, boolean b) {
        return a == b;
    }

    public static void main(String[] args) {
        Scanner _scanner = new Scanner(System.in);
        long i = (0);
        long count = (0);
        long temp = (0);
        long prev = (0);
        long current = (1);
        System.out.println("How many fibonacci numbers to generated");
        count = _scanner.nextLong();
        System.out.println("");
        System.out.println(prev);
        System.out.println(current);
        for (; compareLt((i), (count)); i += (1)) {
            temp = (current);
            current += (prev);
            prev = (temp);
            System.out.println(current);
        }

    }
}