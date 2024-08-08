//
//  ContentView.swift
//  BasketballHighlights
//
//  Created by Swabhan Katkoori on 8/6/24.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack {
            Image(systemName: "basketball")
                .resizable()
                .scaledToFit()
                .frame(width: 75, height: 75)
            Text("Basketball Highlights")
                .dynamicTypeSize(.xxLarge)
        }
        .padding()
        .containerRelativeFrame([.horizontal, .vertical])
        .background(Gradient(colors: [.orange, .cyan]).opacity(0.4))
        
        
    }
    
}

#Preview {
    ContentView()
}
